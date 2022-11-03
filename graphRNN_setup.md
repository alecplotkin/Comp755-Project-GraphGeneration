# Install GraphRNN w/ Nvidia GPU & Ubuntu 22.04 LTS

1. Clone repo and navigate to the GraphRNN submodule directory (GraphRNN/)
2. Install Nvidia drivers and Nvidia CUDA tools via `sudo apt update && sudo apt install nvidia-driver-520 cuda-11-8 cudatoolkit` and reboot before continuing
3. Install python3 and pip w/ `sudo apt install python3 python3-pip`
4. Install the pip requirements w/ `pip install -r requirments.txt`
5. Install cudatoolkit from the pytorch channel using anaconda w/ `conda install cudatoolkit=11.3 -c pytorch`
6. Reboot system again
7. Run nvidia-smi which should match example_nvidia_smi.png
8. Test run! --> `$python3 main.py`
	* If the result is a CUDA error, ensure your args.py CUDA device ID is the same as your CUDA GPU id
		- You can find the correct CUDA device id w/ `$python -c 'import torch; print(torch.cuda.current_device())'` 
		- Open args.py and set the `self.cuda = ` value to the number from the current_device()
9. Success?

## Tips for debugging
* Use a live python environment `$python3` to test torch functionality w/ `import torch` and then running different `torch.cuda.xxx()` functionality.
* On Windows devices, Nvidia has two separate drivers: Game-Ready and Studio-Ready. Only the Studio-Ready driver contains the CUDA packages
* Ensure that the CUDA and nvidia modules are loaded on Linux with `sudo lsmod`
