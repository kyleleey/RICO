import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from torch.autograd import Function


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            per_image_code = False
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        self.per_image_code = per_image_code
        if self.per_image_code:
            # nerf in the wild parameter
            # parameters
            # maximum 1024 images
            self.embeddings = nn.Parameter(torch.empty(1024, 32))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            dims[0] += 32
        
        print("rendering network architecture:")
        print(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, indices):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        
        if self.per_image_code:
            image_code = self.embeddings[indices].expand(rendering_input.shape[0], -1)
            rendering_input = torch.cat([rendering_input, image_code], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.sigmoid(x)
        return x


class SemImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
            sigmoid = 20,
            sigmoid_optim = False
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None

        # MonoSDF, use an all-zero vector as hash output placeholder
        self.grid_feature_dim = 32
        dims[0] += self.grid_feature_dim

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
        
        print("network architecture")
        print(dims)

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.d_out = d_out

        if not sigmoid_optim:
            self.sigmoid_optim = False
            self.sigmoid = sigmoid
        else:
            self.sigmoid_optim = True
            self.sigmoid_basis = nn.Parameter(data=torch.Tensor([np.log(sigmoid)]), requires_grad=True)

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

        self.pool = nn.MaxPool1d(self.d_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        feature = torch.zeros_like(input[:, :1].repeat(1, self.grid_feature_dim))
        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:self.d_out]
        d_output = torch.ones_like(y[:, :1], requires_grad=False, device=y.device)
        g = []
        for idx in range(y.shape[1]):
            gradients = torch.autograd.grad(
                outputs=y[:, idx:idx+1],
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            g.append(gradients)
        
        g = torch.cat(g)
        return g

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]

        if self.sigmoid_optim:
            sigmoid_value = torch.exp(self.sigmoid_basis)
        else:
            sigmoid_value = self.sigmoid

        semantic = sigmoid_value * torch.sigmoid(-sigmoid_value * sdf_raw)
        sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1) # get the minium value of sdf
        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return sdf, feature_vectors, gradients, semantic, sdf_raw

    def get_specific_outputs(self, x, idx):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]

        if self.sigmoid_optim:
            sigmoid_value = torch.exp(self.sigmoid_basis)
        else:
            sigmoid_value = self.sigmoid

        semantic = sigmoid_value * torch.sigmoid(-sigmoid_value * sdf_raw)
        sdf = sdf_raw[:, idx:idx+1]
        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return sdf, feature_vectors, gradients, semantic, output[:,:self.d_out]

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:self.d_out]
        sdf = -self.pool(-sdf.unsqueeze(1)).squeeze(-1)
        return sdf
    
    def get_sdf_vals_and_sdfs(self, x):
        sdf = self.forward(x)[:,:self.d_out]
        sdf_all = sdf
        sdf = -self.pool(-sdf.unsqueeze(1)).squeeze(-1)
        return sdf, sdf_all

    def get_specific_sdf_vals(self, x, idx):
        sdf = self.forward(x)[:,:self.d_out]
        sdf = sdf[:, idx: idx+1]
        return sdf


class RICONetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()

        self.implicit_network = SemImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        
        # NeuS rendering related
        variance_init = conf.get_float('density.variance_init', default=0.05)
        speed_factor = conf.get_float('density.speed_factor', default=10.0)
        self.ln_s = nn.Parameter(data=torch.Tensor([-np.log(variance_init) / speed_factor]), requires_grad=True)
        self.speed_factor = speed_factor

        self.take_sphere_intersection = conf.get_bool('ray_sampler.take_sphere_intersection', default=True)
        self.N_samples = conf.get_int('ray_sampler.N_samples', default=64)
        self.N_samples_extra = conf.get_int('ray_sampler.N_samples_extra', default=32)
        self.N_upsample_iters = conf.get_int('ray_sampler.N_upsample_iters', default=4)

        self.near = conf.get_float('ray_sampler.near', default=0.0)
        self.far = 2.0 * self.scene_bounding_sphere * 1.75

        self.num_semantic = conf.get_int('implicit_network.d_out')
        
        self.render_bg = conf.get_bool('render_bg', default=False)
        self.render_bg_iter = conf.get_int('render_bg_iter', default=1)
    
    def get_s_value(self):
        s = torch.exp(self.ln_s * self.speed_factor)
        return s

    def ray_marching_surface(self, ray_dirs, cam_loc, idx=0, n_steps=128, n_secant_steps=8, max_points=3500000):
        '''
        ray_dirs: [N, 3]
        According to indoor scene initialization, sdf should be first > 0 then < 0
        '''
        ray0 = cam_loc.unsqueeze(0)
        ray_direction = ray_dirs.unsqueeze(0)

        batch_size, n_pts, D = ray0.shape
        device = ray_dirs.device

        
        if not self.take_sphere_intersection:
            near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0], 1).cuda()
        else:
            _, far = self.near_far_from_cube(cam_loc, ray_dirs, bound=self.scene_bounding_sphere)
            near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()

        near = near.reshape(batch_size, n_pts, 1, 1).to(device)
        far = far.reshape(batch_size, n_pts, 1, 1).to(device)

        d_proposal = torch.linspace(0, 1, steps=n_steps).view(1, 1, n_steps, 1).to(device)
        d_proposal = near * (1. - d_proposal) + far * d_proposal

        # [1, n_pts, n_steps, 3]
        p_proposal = ray0.unsqueeze(2).repeat(1, 1, n_steps, 1) + ray_direction.unsqueeze(2).repeat(1, 1, n_steps, 1) * d_proposal
        with torch.no_grad():
            if idx >= 0:
                val = torch.cat([(
                    self.implicit_network.get_specific_sdf_vals(p_split.view(-1, 3), idx=idx))
                    for p_split in torch.split(
                        p_proposal.reshape(batch_size, -1, 3),
                        int(max_points / batch_size), dim=1)], dim=1).view(
                            batch_size, -1, n_steps)
            else:   # use the minimum value to ray_marching the surface
                val = torch.cat([(
                    self.implicit_network.get_sdf_vals(p_split.view(-1, 3)))
                    for p_split in torch.split(
                        p_proposal.reshape(batch_size, -1, 3),
                        int(max_points / batch_size), dim=1)], dim=1).view(
                            batch_size, -1, n_steps)
        
        # val: [1, n_pts, n_steps]
        # Create mask for valid points where the first point is not occupied
        mask_0_not_occupied = val[:, :, 0] > 0

        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]),
                                 torch.ones(batch_size, n_pts, 1).to(device)],
                                dim=-1)
        cost_matrix = sign_matrix * torch.arange(
            n_steps, 0, -1).float().to(device)

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_pos_to_neg = val[torch.arange(batch_size).unsqueeze(-1),
                              torch.arange(n_pts).unsqueeze(-0), indices] > 0

        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_pos_to_neg & mask_0_not_occupied 

        # Get depth values and function values for the interval
        # to which we want to apply the Secant method
        n = batch_size * n_pts
        d_low = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
            batch_size, n_pts)[mask]
        indices = torch.clamp(indices + 1, max=n_steps-1)
        d_high = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_high = val.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]

        ray0_masked = ray0[mask]
        ray_direction_masked = ray_direction[mask]
        
        # Apply surface depth refinement step (e.g. Secant method)
        d_pred = self.secant(
            f_low, f_high, d_low, d_high, n_secant_steps, ray0_masked,
            ray_direction_masked, tau=0., idx=idx)

        # for sanity
        d_pred_out = torch.ones(batch_size, n_pts).to(device)
        d_pred_out[mask] = d_pred
        d_pred_out[mask == 0] = far.reshape(1, -1)[mask == 0]
        d_pred_out[mask_0_not_occupied == 0] = far.reshape(1, -1)[mask_0_not_occupied == 0]

        d_pred_out = d_pred_out.reshape(-1, 1)
        return d_pred_out

    def secant(self, f_low, f_high, d_low, d_high, n_secant_steps,
                          ray0_masked, ray_direction_masked, tau, idx=0):
        ''' Runs the secant method for interval [d_low, d_high].
        Args:
            d_low (tensor): start values for the interval, actually for sdf this should be > 0
            d_high (tensor): end values for the interval, actually for sdf this should be < 0
            n_secant_steps (int): number of steps
            ray0_masked (tensor): masked ray start points
            ray_direction_masked (tensor): masked ray direction vectors
            model (nn.Module): model model to evaluate point occupancies
            c (tensor): latent conditioned code c
            tau (float): threshold value in logits
        '''
        # this guarantee a valid depth no matter f_low > 0 or f_low < 0
        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        for i in range(n_secant_steps):
            p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
            with torch.no_grad():
                if idx >= 0:
                    f_mid = self.implicit_network.get_specific_sdf_vals(p_mid, idx=idx)[...,0] - tau
                else:
                    f_mid = self.implicit_network.get_sdf_vals(p_mid)[...,0] - tau
            
            # ind_low = f_mid < 0
            # ind_low = ind_low
            ind_low = torch.sign(f_mid * f_low)
            ind_low[ind_low <= 0] = 0
            ind_low = ind_low.long()

            if ind_low.sum() > 0:
                d_low[ind_low == 1] = d_pred[ind_low == 1]
                f_low[ind_low == 1] = f_mid[ind_low == 1]
            if (ind_low == 0).sum() > 0:
                d_high[ind_low == 0] = d_pred[ind_low == 0]
                f_high[ind_low == 0] = f_mid[ind_low == 0]

            d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        return d_pred
    
    def near_far_from_cube(self, rays_o, rays_d, bound):
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=self.near)
        far = torch.clamp(far, max=self.far)
        return near, far
    
    def neus_sample(self, cam_loc, ray_dirs, idx=-1):
        '''
        cam_loc: [N, 3]
        ray_dirs: [N, 3]
        '''
        device = cam_loc.device
        perturb = self.training

        if not self.take_sphere_intersection:
            near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0], 1).cuda()
        else:
            _, far = self.near_far_from_cube(cam_loc, ray_dirs, bound=self.scene_bounding_sphere)
            near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()
        
        _t = torch.linspace(0, 1, self.N_samples).float().to(device)
        z_vals = near * (1 - _t) + far * _t

        if perturb:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)

            z_vals = lower + (upper - lower) * t_rand

        with torch.no_grad():
            _z = z_vals     # [N, 64]

            # follow the objsdf setting and use min sdf for sample
            _pts = cam_loc.unsqueeze(-2) + _z.unsqueeze(-1) * ray_dirs.unsqueeze(-2)
            N_rays, N_steps = _pts.shape[0], _pts.shape[1]

            if idx >=0 and idx < self.implicit_network.d_out:
                _sdf = self.implicit_network.get_specific_sdf_vals(_pts.reshape(-1, 3), idx=idx)
                use_defined_idx = True
            else:
                _sdf = self.implicit_network.get_sdf_vals(_pts.reshape(-1, 3))
                use_defined_idx = False
            
            _sdf = _sdf.reshape(N_rays, N_steps)

            for i in range(self.N_upsample_iters):
                prev_sdf, next_sdf = _sdf[..., :-1], _sdf[..., 1:]
                prev_z_vals, next_z_vals = _z[..., :-1], _z[..., 1:]
                mid_sdf = (prev_sdf + next_sdf) * 0.5
                dot_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
                prev_dot_val = torch.cat([torch.zeros_like(dot_val[..., :1], device=device), dot_val[..., :-1]], dim=-1)
                dot_val = torch.stack([prev_dot_val, dot_val], dim=-1)  
                dot_val, _ = torch.min(dot_val, dim=-1, keepdim=False)
                dot_val = dot_val.clamp(-10.0, 0.0)
                
                dist = (next_z_vals - prev_z_vals)
                prev_esti_sdf = mid_sdf - dot_val * dist * 0.5
                next_esti_sdf = mid_sdf + dot_val * dist * 0.5
                
                prev_cdf = cdf_Phi_s(prev_esti_sdf, 64 * (2**i))
                next_cdf = cdf_Phi_s(next_esti_sdf, 64 * (2**i))
                alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
                _w = alpha_to_w(alpha)
                z_fine = rend_util.sample_pdf(_z, _w, self.N_samples_extra // self.N_upsample_iters, det=not perturb)
                _z = torch.cat([_z, z_fine], dim=-1)
                
                _pts_fine = cam_loc.unsqueeze(-2) + z_fine.unsqueeze(-1) * ray_dirs.unsqueeze(-2)
                N_rays, N_steps_fine = _pts_fine.shape[0], _pts_fine.shape[1]

                if use_defined_idx:
                    sdf_fine = self.implicit_network.get_specific_sdf_vals(_pts_fine.reshape(-1, 3), idx=idx)
                else:
                    sdf_fine = self.implicit_network.get_sdf_vals(_pts_fine.reshape(-1, 3))

                sdf_fine = sdf_fine.reshape(N_rays, N_steps_fine)
                _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
                _z, z_sort_indices = torch.sort(_z, dim=-1)
                
                _sdf = torch.gather(_sdf, 1, z_sort_indices)
            
            z_all = _z
        return z_all

    def forward(self, input, indices, iter_step=-1):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
        depth_scale = ray_dirs_tmp[0, :, 2:]  # [N, 1]

        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        '''
        NeuS sample
        '''
        
        z_vals = self.neus_sample(cam_loc, ray_dirs)
        N_samples_tmp = z_vals.shape[1]

        '''
        1. Weight computation with sampled points
        '''

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)  # [N_rays, N_samples_tmp, 3]
        points_flat_tmp = points.reshape(-1, 3)

        sdf_tmp, sdf_all_tmp = self.implicit_network.get_sdf_vals_and_sdfs(points_flat_tmp)
        sdf_tmp = sdf_tmp.reshape(-1, N_samples_tmp)
        s_value = self.get_s_value()

        cdf, opacity_alpha = sdf_to_alpha(sdf_tmp, s_value)     # [N_rays, N_samples_tmp-1]

        sdf_all_tmp = sdf_all_tmp.reshape(-1, N_samples_tmp, self.num_semantic)

        '''
        2. Forward with mid-points
        '''

        z_mid_vals = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        N_samples = z_mid_vals.shape[1]

        points_mid = cam_loc.unsqueeze(1) + z_mid_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)  # [N_rays, N_samples, 3]
        points_flat = points_mid.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)
        
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_outputs(points_flat)

        # acquire the sdfs where min_sdf < 0
        min_sdf, min_idx = sdf_raw.min(dim=-1, keepdim=True) # [M, 1]
        min_mask = min_sdf < 0
        min_mask = min_mask.repeat(1, sdf_raw.shape[-1])     # [M, num_semantic]
        min_sdf_mask = sdf_raw == min_sdf                    # [M, num_semantic]
        valid_not_min_mask = min_mask & ~min_sdf_mask
        valid_other_sdfs = sdf_raw[valid_not_min_mask]

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, indices)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        semantic = semantic.reshape(-1, N_samples, self.num_semantic)

        '''
        3. Volume rendering
        '''
        weights = alpha_to_w(opacity_alpha)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1)*semantic, 1)

        raw_depth_values = torch.sum(weights * z_mid_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * raw_depth_values

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)


        output = {
            'rgb_values': rgb_values,
            'semantic_values': semantic_values,
            'depth_values': depth_values,
            'z_vals': z_mid_vals,
            'depth_vals': z_mid_vals * depth_scale,
            'sdf': sdf.reshape(z_mid_vals.shape),
            'weights': weights,
            'other_sdf': valid_other_sdfs,
            'semantic': semantic,
        }

        if self.training:
            output['grad_theta'] = gradients

            '''
            Object Point-SDF Loss, L_op
            '''
            surf_bg_z_vals = self.ray_marching_surface(ray_dirs, cam_loc, idx=0) # [N, 1]
            # the sdf value of objects that behind bg surface
            bg_surf_back_mask = z_mid_vals > surf_bg_z_vals # [1024, 98]
            sdf_all = sdf_raw.reshape(z_mid_vals.shape[0], z_mid_vals.shape[1], -1)
            objs_sdfs_bg_back = sdf_all[bg_surf_back_mask][..., 1:]  # [K, num_semantics-1]

            output['obj_sdfs_behind_bg'] = objs_sdfs_bg_back

            semantic_map = torch.argmax(semantic_values, dim=-1)   # [1024]

            valid_indices = torch.unique(semantic_map).int().tolist()
            if 0 in valid_indices:
                valid_indices.remove(0)
            
            '''
            Patch-based Background Smoothness, L_bs
            '''
            iter_check = iter_step % self.render_bg_iter
            if self.render_bg and iter_check == 0:
                bg_render_input = {}
                bg_render_input["pose"] = input["pose"]
                bg_render_input["intrinsics"] = input["intrinsics"]

                # construct patch uv
                patch_size = 32
                n_patches = 1

                x0 = np.random.randint(0, 384 - patch_size + 1, size=(n_patches, 1, 1))
                y0 = np.random.randint(0, 384 - patch_size + 1, size=(n_patches, 1, 1))
                xy0 = np.concatenate([x0, y0], axis=-1)
                patch_idx = xy0 + np.stack(np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),axis=-1).reshape(1, -1, 2)
                uv0 = torch.from_numpy(patch_idx).float().reshape(1, -1, 2).float().cuda()
                bg_render_input["uv"] = uv0

                background_render = self.render_via_semantic(bg_render_input, indices, idx=0, render_rgb=False)
                output['background_render'] = background_render
            else:
                output['background_render'] = None
            
            '''
            Reversed Depth Loss, L_rd
            '''

            r_obj_depths = []
            r_bg_depths = []

            for valid_ind in valid_indices:
                obj_mask = semantic_map == valid_ind
                bg_vals = surf_bg_z_vals[obj_mask]  # [K, 1]

                reverse_z_mid_vals = z_mid_vals[obj_mask] # [t1, t2, t3,..., tn]
                sum_value = reverse_z_mid_vals[:, 0] + reverse_z_mid_vals[:, -1]

                r_bg_vals = sum_value.unsqueeze(-1) - bg_vals

                sum_value = sum_value.unsqueeze(-1).repeat(1, reverse_z_mid_vals.shape[-1])
                r_z_mid_vals = sum_value - reverse_z_mid_vals
                r_z_mid_vals = torch.flip(r_z_mid_vals, [-1])

                r_obj_sdfs = sdf_all_tmp[obj_mask]
                r_obj_sdfs = r_obj_sdfs[..., valid_ind]     # [K, 96] in right order
                r_obj_sdfs = torch.flip(r_obj_sdfs, [-1])   # in reverse order
                
                r_mask = r_obj_sdfs[:, 0] > 0.

                if r_mask.sum() == 0:
                    continue
                else:
                    r_obj_sdfs = r_obj_sdfs[r_mask]
                    r_z_mid_vals = r_z_mid_vals[r_mask]
                    r_bg_vals = r_bg_vals[r_mask]

                _, r_opac = sdf_to_alpha(r_obj_sdfs, s_value)
                r_weights = alpha_to_w(r_opac)

                r_d_vals = torch.sum(r_weights * r_z_mid_vals, 1, keepdims=True) / (r_weights.sum(dim=1, keepdims=True) +1e-8)

                r_obj_depths.append(r_d_vals)
                r_bg_depths.append(r_bg_vals)
            
            if len(r_obj_depths) > 0:
                r_obj_depths = torch.cat(r_obj_depths, 0)
                r_bg_depths = torch.cat(r_bg_depths, 0)
            output['obj_d_vals'] = r_obj_depths
            output['bg_d_vals'] = r_bg_depths

        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        
        # transform to local coordinate system
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()
        
        output['normal_map'] = normal_map

        return output

    def render_via_semantic(self, input, indices, idx=0, render_rgb=False):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"] 
        # pose = fix_rot

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
        depth_scale = ray_dirs_tmp[0, :, 2:]  # [N, 1]

        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        '''
        NeuS sample
        '''
        
        z_vals = self.neus_sample(cam_loc, ray_dirs, idx=idx)
        N_samples_tmp = z_vals.shape[1]

        '''
        1. Weight computation with sampled points
        '''

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)  # [N_rays, N_samples_tmp, 3]
        points_flat_tmp = points.reshape(-1, 3)

        scene_sdf, sdf_all = self.implicit_network.get_sdf_vals_and_sdfs(points_flat_tmp)
        scene_sdf = scene_sdf.reshape(-1, N_samples_tmp)

        sdf_tmp = sdf_all[..., idx:idx+1]
        sdf_tmp = sdf_tmp.reshape(-1, N_samples_tmp)
        s_value = self.get_s_value()

        cdf, opacity_alpha = sdf_to_alpha(sdf_tmp, s_value)     # [N_rays, N_samples_tmp-1]
        _, scene_opacity_alpha = sdf_to_alpha(scene_sdf, s_value)

        '''
        2. Forward with mid-points
        '''

        z_mid_vals = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        N_samples = z_mid_vals.shape[1]

        points_mid = cam_loc.unsqueeze(1) + z_mid_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)  # [N_rays, N_samples, 3]
        points_flat = points_mid.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)
        
        # gradients here are get from queried idx-sdf
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_specific_outputs(points_flat, idx=idx)

        weights = alpha_to_w(opacity_alpha)
        scene_weights = alpha_to_w(scene_opacity_alpha)

        if render_rgb:
            rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, indices)
            rgb = rgb_flat.reshape(-1, N_samples, 3)
            rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        else:
            rgb = None
            rgb_values = None

        semantic = semantic.reshape(-1, N_samples, self.num_semantic)

        '''
        3. Volume rendering
        '''
        semantic_values = torch.sum(scene_weights.unsqueeze(-1)*semantic, 1)

        depth_values = torch.sum(weights * z_mid_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb_values': rgb_values,
            'semantic_values': semantic_values,
            'depth_values': depth_values,
        }

        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        
        # transform to local coordinate system
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()
        
        output['normal_map'] = normal_map

        return output



def cdf_Phi_s(x, s):
    # den = 1 + torch.exp(-s*x)
    # y = 1./den
    # return y
    return torch.sigmoid(x*s)


def sdf_to_alpha(sdf: torch.Tensor, s):
    # [(B), N_rays, N_pts]
    cdf = cdf_Phi_s(sdf, s)
    # [(B), N_rays, N_pts-1]
    # TODO: check sanity.
    opacity_alpha = (cdf[..., :-1] - cdf[..., 1:]) / (cdf[..., :-1] + 1e-10)
    opacity_alpha = torch.clamp_min(opacity_alpha, 0)
    return cdf, opacity_alpha


def sdf_to_w(sdf: torch.Tensor, s):
    device = sdf.device
    # [(B), N_rays, N_pts-1]
    cdf, opacity_alpha = sdf_to_alpha(sdf, s)

    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
            1.0 - opacity_alpha + 1e-10,
        ], dim=-1)
    
    # [(B), N_rays, N_pts-1]
    visibility_weights = opacity_alpha *\
        torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return cdf, opacity_alpha, visibility_weights


def alpha_to_w(alpha: torch.Tensor):
    device = alpha.device
    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*alpha.shape[:-1], 1], device=device),
            1.0 - alpha + 1e-10,
        ], dim=-1)
    
    # [(B), N_rays, N_pts-1]
    visibility_weights = alpha *\
        torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return visibility_weights