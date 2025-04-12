import numpy as np
import valmetrics as vm
import matplotlib.pyplot as plt

def main() -> None:
    num_exp = 4
    num_sim = 2
    num_diff = num_exp-num_sim
    exp_data = np.arange(0,num_exp)
    sim_data = exp_data[num_diff:]+num_exp/2

    mavm_res = vm.mavm(sim_data,exp_data)

    vm.mavm_figs(mavm_res,
                f"TEST",
                "field",
                save_tag="TEST")
    plt.show()


if __name__ == "__main__":
    main()