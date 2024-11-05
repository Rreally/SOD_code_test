import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sph_utils import xy2angle, pruned_inf, to_3dsphere, get_face
from utils.sph_utils import face_to_cube_coord, norm_to_cube

class Cube2Equi(nn.Module):
    def __init__(self):
        super(Cube2Equi, self).__init__()
        self.scale_c = 1

    def _config(self, input_w):
        in_width = input_w * self.scale_c
        out_w = in_width * 4
        out_h = in_width * 2

        face_map = torch.zeros((out_h, out_w))

        YY, XX = torch.meshgrid(torch.arange(out_h), torch.arange(out_w))

        theta, phi = xy2angle(XX, YY, out_w, out_h)
        theta = pruned_inf(theta)
        phi = pruned_inf(phi)

        _x, _y, _z = to_3dsphere(theta, phi, 1)
        face_map = get_face(_x, _y, _z, face_map)
        x_o, y_o = face_to_cube_coord(face_map, _x, _y, _z)

        out_coord = torch.stack([x_o, y_o], dim=2) 
        out_coord = norm_to_cube(out_coord, in_width)

        return out_coord, face_map

    def forward(self, input_data):
        warpout_list = []
        for input in torch.split(input_data, 1 , dim=0):
            inputdata = input.squeeze(dim=0).clone().detach().cuda()
            assert inputdata.size()[2] == inputdata.size()[3]
            gridf, face_map = self._config(inputdata.size()[2])
            gridf = gridf.clone().detach().cuda()
            face_map = face_map.clone().detach().cuda()
            out_w = int(gridf.size(1))
            out_h = int(gridf.size(0))
            in_width = out_w/4
            depth = inputdata.size(1)
            warp_out = torch.zeros([1, depth, out_h, out_w], dtype=torch.float32).clone().detach().requires_grad_().cuda()
            gridf = (gridf-torch.max(gridf)/2)/(torch.max(gridf)/2)

            for f_idx in range(0, 6):
                face_mask = face_map == f_idx
                expanded_face_mask = face_mask.expand(1, inputdata.size(1), face_mask.size(0), face_mask.size(1))
                warp_out[expanded_face_mask] = nn.functional.grid_sample(torch.unsqueeze(inputdata[f_idx], 0), torch.unsqueeze(gridf, 0), align_corners=True)[expanded_face_mask]

            warpout_list.append(warp_out)
        return torch.cat(warpout_list, dim=0)


class C2EB(nn.Module):
    def __init__(self):
        super(C2EB, self).__init__()
        self.projection = Cube2Equi()

    def forward(self, cubefeature_list):
        cubefeatures = torch.stack(cubefeature_list, dim=1)
        equi_feature = self.projection(cubefeatures)
        return equi_feature