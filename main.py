from algorithm import PPO
from models import Policy_layer, Action_layer
from scripts.Town05 import CarlaEnv_T5
import torch.optim as optim
import torch

max_step = 1000

def main():
    env = CarlaEnv_T5()
    uplayer = Policy_layer()
    dolayer = Action_layer()

    up_optimizer = optim.Adam(uplayer.parameters(), lr=0.001)
    up_ppo = PPO(uplayer, up_optimizer, dolayer)
    do_optimizer = optim.Adam(dolayer.parameters(), lr=0.001)
    do_ppo = PPO(dolayer, do_optimizer, dolayer)

    while True:
        up_states, down_states = env.reset()
        ulog_probs = [], uvalues = [], ustates = [], uactions = [], urewards = [], umasks = []
        dlog_probs = [], dvalues = [], dstates = [], dactions = [], drewards = [], dmasks = []
        for tick_num in range(max_step):
            if tick_num % 10 == 0: 
                up_output, ulog_prob, uvalue = uplayer(up_states)
                mode = "all"
            else: mode = "down"
            action, dlog_prob, dvalue = dolayer(down_states)
            next_down_states, dreward, ddone, next_up_states, ureward, udone = env.step(action, up_output, mode=mode)
            dlog_probs.append(dlog_prob), dvalues.append(dvalue), dstates.append(down_states), dactions.append(action)
            drewards.append(torch.tensor([dreward], dtype=torch.float32)), dmasks.append(torch.tensor([1-ddone], dtype=torch.float32))
            if mode == "all":
                ulog_probs.append(ulog_prob), uvalues.append(uvalue), ustates.append(up_states), uactions.append(up_output)
                urewards.append(torch.tensor([ureward], dtype=torch.float32)), umasks.append(torch.tensor([1-udone], dtype=torch.float32))
            if ddone: break
            up_states = next_up_states, down_states = next_down_states
        
        _, _, unext_value = uplayer(next_up_states)
        ureturns = up_ppo.compute_returns(unext_value, urewards, umasks)
        ustates = torch.cat(ustates, dim=0)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns)
        values = torch.cat(values)
        uadvantages = ureturns - uvalues
        up_ppo.ppo_update(uplayer, up_optimizer, 4, 64, ustates, uactions, ulog_probs, ureturns, uadvantages)
        do_ppo.update(down_states, action, dreward, ddone)

if __name__ == '__main__':
    main()