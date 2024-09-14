from config import CONFIG
import pandapower as pp
import pandapower.networks as pn
import torch

def solve_dcopf_ieee14():
    net = pn.case14()
    net.gen['controllable'] = True
    if 'poly_cost' in net:
        net.poly_cost.drop(net.poly_cost.index, inplace=True)
    
    for i in net.gen.index:
        pp.create_poly_cost(net, element=i, et='gen', cp1_eur_per_mw=CONFIG['dcopf_cost_factor']+i*CONFIG['dcopf_cost_factor'], cp0_eur=0)
    
    try:
        pp.rundcopp(net)
        print("DCOPF completed successfully")
    except pp.powerflow.LoadflowNotConverged:
        print("DCOPF did not converge")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    
    return net

def extract_dcopf_features(net):
    gen_p = torch.tensor(net.res_gen.p_mw.values, dtype=torch.float)
    bus_va = torch.tensor(net.res_bus.va_degree.values, dtype=torch.float)
    line_p = torch.tensor(net.res_line.p_from_mw.values, dtype=torch.float)
    return gen_p, bus_va, line_p