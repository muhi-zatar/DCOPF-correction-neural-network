import torch
import torch.nn.functional as F
from utils.dcopf_solver import solve_dcopf_ieee14, extract_dcopf_features

def prepare_data(dataset, device):
    processed_data = []
    dcopf_net = solve_dcopf_ieee14()
    
    if dcopf_net is None:
        raise ValueError("DCOPF could not be solved for the IEEE 14-bus system")
    
    dcopf_gen_p, dcopf_bus_va, dcopf_line_p = extract_dcopf_features(dcopf_net)
    
    for data in dataset:
        data = data.to(device)
        
        # Combine ACOPF inputs with DCOPF outputs
        num_generators = data['generator'].num_nodes
        num_buses = data['bus'].num_nodes
        num_branches = data['branch'].num_edges

        gen_p_padded = F.pad(dcopf_gen_p, (0, max(0, num_generators - len(dcopf_gen_p))))[:num_generators].to(device)
        bus_va_padded = F.pad(dcopf_bus_va, (0, max(0, num_buses - len(dcopf_bus_va))))[:num_buses].to(device)
        line_p_padded = F.pad(dcopf_line_p, (0, max(0, num_branches - len(dcopf_line_p))))[:num_branches].to(device)

        data['generator'].x = torch.cat([data['generator'].x, gen_p_padded.unsqueeze(1)], dim=1)
        data['bus'].x = torch.cat([data['bus'].x, bus_va_padded.unsqueeze(1)], dim=1)
        data['branch'].edge_attr = torch.cat([data['branch'].edge_attr, line_p_padded.unsqueeze(1)], dim=1)

        processed_data.append(data)
    
    return processed_data
