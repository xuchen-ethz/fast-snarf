import tinycudann as tcnn
import torch

""" MLP for neural implicit shapes. The code is based on https://github.com/lioryariv/idr with adaption. """
class ImplicitNetwork(torch.nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        width,
        depth,
        geometric_init=True,
        bias=1.0,
        weight_norm=True,
        multires=0,
        skip_layer=[],
        cond_layer=[],
        cond_dim=0,
        dim_cond_embed=-1,
        use_tanh=False
    ):
        super().__init__()


        self.cond_dim = cond_dim
        config_encoding = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 15,
            "base_resolution": 32,
            "per_level_scale": 1.5
        }

        config_encoding = {
            "otype": "Grid",
            "type": "Dense",
            "n_levels": 1,
            "n_features_per_level": 8,
            "base_resolution": 64,
        }

        # config_network = {
        #     "otype": "FullyFusedMLP",
        #     "activation": "ReLU",
        #     "output_activation": "None",
        #     "n_neurons": 64,
        #     "n_hidden_layers": 2
        # }
        # config_encoding = {
        #     "otype": "Frequency",
        #     "n_frequencies": 4
        # }

        config_network = {
	        "otype": "CutlassMLP", 
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 256,
            "n_hidden_layers": 8
        }
        self.encoding = tcnn.Encoding(d_in, config_encoding)

        self.model = tcnn.Network(self.encoding.n_output_dims+cond_dim, d_out, config_network)
        
        # self.encoding = tcnn.Encoding(d_in, config_encoding)

        # self.model = tcnn.Network(self.encoding.n_output_dims+cond_dim, d_out, config_network)
        
    def forward(self, input, cond, mask=None):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            N: number of points
            D: input dimension
            
        Args:
            input (tensor): network input. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor, optional): only masked inputs are fed into the network. shape: [B, N]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [N, D, ?]
        """


        n_batch, n_point, n_dim = input.shape

        if n_batch * n_point == 0:
            return input

        in_bbox = ((input>0) & (input<1)).all(-1)
        # reshape to [N,?]
        input = input.reshape(n_batch * n_point, n_dim)
        if mask is not None:
            input = input[mask]

        input = self.encoding(input)
        if self.cond_dim>0:
            cond = cond["smpl"]
            n_batch, n_cond = cond.shape
            input_cond = cond.unsqueeze(1).expand(n_batch, n_point, n_cond)
            input_cond = input_cond.reshape(n_batch * n_point, n_cond)

            if mask is not None:
                input_cond = input_cond[mask]

            input = torch.cat([input, input_cond], dim=-1)

        x = self.model(input).float()


        # add placeholder for masked prediction
        if mask is not None:
            x_full = torch.zeros(n_batch * n_point, x.shape[-1], device=x.device)
            x_full[mask] = x
        else:
            x_full = x

        x_full = x_full.reshape(n_batch, n_point, -1)
        x_full[~in_bbox,:] = -1000
        return x_full