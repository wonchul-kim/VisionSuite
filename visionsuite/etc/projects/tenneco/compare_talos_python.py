import os 
import os.path as osp 
from glob import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm

# orders = ['1st', '2nd']
orders = ['1st']
cases = ['diff_from_talos', 'diff_from_python']
model = 'mask2former'
# model = 'deeplabv3plus'

height, width, channel = 768, 1120, 4
rtol = 5e-2
atol = 5e-2
ratio_threshold = 0 # %

for order in orders:
    for case in cases:
        num_same, num_not_same = 0, 0
        talos_input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/talos/talos/{model}/{order}/{case}'       
        if model == 'deeplabv3plus':
            python_input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/talos/python/{model}/test/{order}/{case}/vis/raw'
        else:
            python_input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/talos/python/{model}/test/{order}/{case}/raw'
        output_dir = f'/DeepLearning/etc/_athena_tests/benchmark/talos/compare/{model}/ratio{ratio_threshold}_rtol{rtol}_atol{atol}/{order}/{case}'

        if not osp.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        bin_files = glob(osp.join(talos_input_dir, '*.bin'))
        # bin_files = glob(osp.join(talos_input_dir, '125020717061509_7_Outer_1.bmp.bmp.bin'))
        if model != 'deeplabv3plus':
            npz_files = glob(osp.join(python_input_dir, '*.npy'))

        for bin_file in tqdm(bin_files, desc=f"order-{order}-case-{case}"):
            filename = osp.split(osp.splitext(bin_file)[0])[-1].split(".")[0]
            if model == 'deeplabv3plus':
                npz_file = osp.join(python_input_dir, filename + '.npz')
            else:
                _filename = [f for f in npz_files if osp.split(osp.splitext(f)[0])[-1].startswith(filename)]                    
                assert len(_filename) == 1, ValueError(f"There are more than 2 filenames: {_filename}")
                npz_file = _filename[0]
            
            if not osp.exists(npz_file):
                print(f'There is no such npz file: {npz_file}')
                continue
            
            
            bin_arr = np.fromfile(bin_file, dtype=np.float32).reshape((channel, height, width))
            bin_arr = np.transpose(bin_arr, (1, 2, 0))
            npz_arr = np.load(npz_file)
            if 'arr' in npz_arr:
                npz_arr = np.load(npz_file)['arr']
            if npz_arr.shape == (channel, height, width):
                npz_arr = np.transpose(npz_arr, (1, 2, 0))
                
            
            assert bin_arr.shape == npz_arr.shape # hwc
            assert bin_arr.dtype == npz_arr.dtype # float32          
            
            is_close = np.allclose(bin_arr, npz_arr, rtol=rtol, atol=atol)
            is_same = is_close

            diff = np.abs(bin_arr - npz_arr)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            ratios = []
            for ch in range(bin_arr.shape[-1]):
                ratio = np.sum(np.abs(bin_arr[..., ch] - npz_arr[..., ch]) > rtol)/diff.size*100
                ratios.append(ratio)
                
                if not is_close and ratio > ratio_threshold:
                    is_same = False
                
            _output_dir = osp.join(output_dir, str(is_same))
            if not osp.exists(_output_dir):
                os.mkdir(_output_dir)
                
                
            diff_mask = ~np.isclose(bin_arr, npz_arr, rtol=rtol, atol=atol)
            channels = diff_mask.shape[2]
            fig, axes = plt.subplots(1, channels, figsize=(5 * channels, 10))

            # First row: diff_talos_python
            for c in range(channels):
                ax = axes[c] if channels > 1 else axes[0]
                ax.imshow(diff_mask[..., c])
                ax.set_title(f"TALOS - Python (Channel {c})")
                ax.axis("off")

            plt.suptitle("Difference Maps per Channel", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(osp.join(_output_dir, filename + '_diff_mask.png'))
            plt.close()
                
                
            # pred images ------------------------------------------------------------
            talos_pred_img = cv2.imread(osp.join(talos_input_dir, filename + '.bmp.bmp.png'))
            if talos_pred_img is None:
                talos_pred_img = cv2.imread(osp.join(talos_input_dir, filename + '.bmp.bmp'))
            if model == 'deeplabv3plus':
                python_pred_img = cv2.imread(osp.join(python_input_dir, '../' + filename + '_argmax.png'))
            else:
                python_pred_img = cv2.imread(osp.join(python_input_dir, '../vis/' + osp.split(osp.splitext(npz_file)[0])[-1] + '.png'))
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(talos_pred_img, cmap='gray')
            axes[0].axis("off")

            axes[1].imshow(python_pred_img, cmap='gray')
            axes[1].axis("off")

            plt.suptitle(f"talos (left) vs. python (right)", fontsize=14)
            plt.tight_layout()
            plt.savefig(osp.join(_output_dir, filename + '_pred.png'))
            plt.close()
            
            # --- 차이 계산 ---
            diff_talos_python = bin_arr - npz_arr
            diff_python_talos = npz_arr - bin_arr
            max_diff = np.max(diff_talos_python)
            mean_diff = np.mean(diff_talos_python)

            # --- 히스토그램 ---
            # First histogram
            fig = plt.figure(figsize=(20, 10))
            plt.subplots_adjust(bottom=0.2)  # 하단 여백을 충분히 줌
            fig.text(0.5, 0.04, f"same = {is_same}, max diff: {max_diff}, mean diff: {mean_diff}, ratios: {ratios}", 
                ha='center', fontsize=10)
            fig.text(0.5, 0.01, f"(rtol = {rtol}, atol = {atol}, ratio_threshold = {ratio_threshold})%", 
                ha='center', fontsize=10)
            plt.subplot(2, 2, 1)
            n, bins, patches = plt.hist(bin_arr.flatten(), bins=100, color='skyblue', edgecolor='k')
            plt.title("TALOS")
            plt.xlabel("Value")
            plt.ylabel("Pixel Count(log)")
            plt.grid(True)
            plt.yscale("log")  # 로그 스케일

            # 막대 위에 개수 표시
            for count, bin_left, patch in zip(n, bins, patches):
                if count > 0:
                    plt.text(bin_left + (bins[1] - bins[0]) / 2, count, f"{int(count)}", 
                            ha='center', va='bottom', fontsize=6, rotation=90)

            # Second histogram
            plt.subplot(2, 2, 2)
            n2, bins2, patches2 = plt.hist(npz_arr.flatten(), bins=100, color='skyblue', edgecolor='k')
            plt.title("Python")
            plt.xlabel("Value")
            plt.ylabel("Pixel Count(log)")
            plt.grid(True)
            plt.yscale("log")  # 로그 스케일
            for count, bin_left, patch in zip(n2, bins2, patches2):
                if count > 0:
                    plt.text(bin_left + (bins2[1] - bins2[0]) / 2, count, f"{int(count)}", 
                            ha='center', va='bottom', fontsize=6, rotation=90)

            plt.subplot(2, 2, 3)
            n, bins, patches = plt.hist(diff_talos_python.flatten(), bins=100, color='skyblue', edgecolor='k')
            plt.title("TALOS - Python")
            plt.xlabel("Absolute Difference")
            plt.ylabel("Pixel Count(log)")
            plt.grid(True)
            plt.yscale("log")  # 로그 스케일

            # 막대 위에 개수 표시
            for count, bin_left, patch in zip(n, bins, patches):
                if count > 0:
                    plt.text(bin_left + (bins[1] - bins[0]) / 2, count, f"{int(count)}", 
                            ha='center', va='bottom', fontsize=6, rotation=90)

            # Second histogram
            plt.subplot(2, 2, 4)
            n2, bins2, patches2 = plt.hist(diff_python_talos.flatten(), bins=100, color='skyblue', edgecolor='k')
            plt.title("Python - TALOS")
            plt.xlabel("Absolute Difference")
            plt.ylabel("Pixel Count(log)")
            plt.grid(True)
            plt.yscale("log")  # 로그 스케일

            # 막대 위에 개수 표시
            for count, bin_left, patch in zip(n2, bins2, patches2):
                if count > 0:
                    plt.text(bin_left + (bins2[1] - bins2[0]) / 2, count, f"{int(count)}", 
                            ha='center', va='bottom', fontsize=6, rotation=90)
            # plt.tight_layout()
            plt.savefig(osp.join(_output_dir, filename + '_histo.jpg'))
            plt.close()

            
            if not is_same:
                num_not_same += 1
                if diff_talos_python.ndim == 3:  # (H, W, C)
                    channels = diff_talos_python.shape[2]

                    # Create a 2-row, N-channel column layout
                    fig, axes = plt.subplots(1, channels, figsize=(5 * channels, 10))

                    # First row: diff_talos_python
                    for c in range(channels):
                        ax = axes[c] if channels > 1 else axes[0]
                        ax.imshow(np.abs(diff_talos_python)[..., c], cmap='hot', interpolation='nearest')
                        ax.set_title(f"TALOS - Python (Channel {c})")
                        ax.axis("off")
                        fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

                    plt.suptitle("Difference Maps per Channel", fontsize=16)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.savefig(osp.join(_output_dir, filename + '_map.png'))
                    plt.close()
                    
                    # Create a 2-row, N-channel column layout
                    fig, axes = plt.subplots(2, channels, figsize=(5 * channels, 10))

                    # First row: diff_talos_python
                    for c in range(channels):
                        ax = axes[0, c] if channels > 1 else axes[0]
                        ax.imshow(bin_arr[..., c], cmap='hot', interpolation='nearest')
                        ax.set_title(f"TALOS (Channel {c})")
                        ax.axis("off")
                        fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

                    # Second row: diff_python_talos
                    for c in range(channels):
                        ax = axes[1, c] if channels > 1 else axes[1]
                        ax.imshow(npz_arr[..., c], cmap='hot', interpolation='nearest')
                        ax.set_title(f"Python (Channel {c})")
                        ax.axis("off")
                        fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

                    plt.suptitle("Raw Prediction per Channel", fontsize=16)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.savefig(osp.join(_output_dir, filename + '_raw.png'))
                    plt.close()
                        
                else:  # 2D일 경우 (H, W)
                    plt.figure(figsize=(6, 5))
                    plt.imshow(diff, cmap='hot', interpolation='nearest')
                    plt.colorbar(label='Absolute Difference')
                    plt.title("Diff Map")
                    plt.tight_layout()
                    plt.savefig(osp.join(_output_dir, filename + '_map.jpg'))
                    plt.close()
            else:
                num_same += 1
                

        num_txt = open(osp.join(output_dir, 'numbers.txt'), 'w')
        num_txt.write(f'number of same: {num_same}\n')
        num_txt.write(f'number of NOT same: {num_not_same}\n')
        num_txt.close()

