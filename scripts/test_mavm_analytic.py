import numpy as np
import valmetrics as vm
import matplotlib.pyplot as plt

def main() -> None:

    # #---------------------------------------------------------------------------
    # # TEST01
    # # More experiments, no remainder, no overlap
    # test_str = "test01"

    # num_exp = 4
    # num_sim = 2
    # num_diff = num_exp-num_sim
    # exp_data = np.arange(0,num_exp)-num_exp/2
    # sim_data = exp_data[num_diff:]+exp_data[-1]+1

    # mavm_res = vm.mavm(sim_data,exp_data,test=test_str)

    # vm.mavm_figs(mavm_res,
    #             test_str,
    #             "field",
    #             save_tag=test_str)

    # #---------------------------------------------------------------------------
    # # TEST02
    # # More simulations, no remainder, no overlap
    # test_str = "test02"
    # num_exp = 2
    # num_sim = 4
    # num_diff = num_sim-num_exp
    # sim_data = np.arange(0,num_sim)-num_sim/2
    # exp_data = sim_data[num_diff:]+sim_data[-1]+2

    # mavm_res = vm.mavm(sim_data,exp_data,test=test_str)

    # vm.mavm_figs(mavm_res,
    #             test_str,
    #             "field",
    #             save_tag=test_str)

    # #---------------------------------------------------------------------------
    # # TEST03
    # # More experiments, no remainder, overlap symmertic
    # test_str = "test03"

    # num_exp = 4
    # num_sim = 2
    # exp_data = np.arange(0,num_exp)-num_exp/2
    # sim_data = np.arange(0,num_sim)-num_sim/2

    # mavm_res = vm.mavm(sim_data,exp_data,test=test_str)

    # vm.mavm_figs(mavm_res,
    #             test_str,
    #             "field",
    #             save_tag=test_str)

    # #---------------------------------------------------------------------------
    # # TEST04
    # # More experiments, remainder, no overlap
    # test_str = "test04"

    # num_exp = 4
    # num_sim = 3
    # num_diff = num_exp-num_sim
    # exp_data = np.arange(0,num_exp)-num_exp/2
    # sim_data = exp_data[num_diff:]+exp_data[-1]+1

    # mavm_res = vm.mavm(sim_data,exp_data,test=test_str)

    # vm.mavm_figs(mavm_res,
    #             test_str,
    #             "field",
    #             save_tag=test_str)


    # #---------------------------------------------------------------------------
    # # TEST05
    # # More simulations, remainder, no overlap
    # test_str = "test05"
    # num_exp = 3
    # num_sim = 4
    # num_diff = num_sim-num_exp
    # sim_data = np.arange(0,num_sim)-num_sim/2
    # exp_data = sim_data[num_diff:]+sim_data[-1]+2

    # mavm_res = vm.mavm(sim_data,exp_data,test=test_str)

    # vm.mavm_figs(mavm_res,
    #             test_str,
    #             "field",
    #             save_tag=test_str)

    # #---------------------------------------------------------------------------
    # # TEST06
    # # More experiments, no remainder, overlap symmertic
    # test_str = "test06"

    # num_exp = 4
    # num_sim = 3
    # exp_data = np.arange(0,num_exp)-num_exp/2
    # sim_data = np.arange(0,num_sim)-num_sim/2

    # mavm_res = vm.mavm(sim_data,exp_data,test=test_str)

    # vm.mavm_figs(mavm_res,
    #             test_str,
    #             "field",
    #             save_tag=test_str)

    # plt.show()

    # #---------------------------------------------------------------------------
    # # TEST07
    # test_str = "test07"

    # num_exp = 100
    # num_sim = 10000
    # offset = num_sim
    # exp_data = np.arange(0,num_exp)-num_exp/2 + 1.25*offset
    # sim_data = np.arange(0,num_sim)-num_sim/2 + offset

    # mavm_res = vm.mavm(sim_data,exp_data,test=test_str)

    # vm.mavm_figs(mavm_res,
    #             test_str,
    #             "field",
    #             save_tag=test_str)

    #---------------------------------------------------------------------------
    # TEST07
    test_str = "test08"

    num_exp = 100
    num_sim = 10000
    #exp_data = np.random.normal(loc=233.0, scale=0.1, size=(num_exp,))
    #exp_data = np.linspace(233.0-1.25,233.0+1.25,num_exp)
    exp_data = np.random.uniform(low=230.95,high=237.16, size=(25,))
    exp_data = np.repeat(exp_data,4)
    sim_data = np.random.normal(loc=238.0, scale=4.0, size=(num_sim,))

    mavm_res = vm.mavm(sim_data,exp_data,test=test_str)

    vm.mavm_figs(mavm_res,
                test_str,
                "field",
                save_tag=test_str)

    print(f"{sim_data.dtype=}")
    print(f"{exp_data.dtype=}")
    print(f"{mavm_res['d+']=}")
    print(f"{mavm_res['d-']=}")

    plt.show()




if __name__ == "__main__":
    main()