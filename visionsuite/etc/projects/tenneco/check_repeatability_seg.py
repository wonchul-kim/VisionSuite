from check_repeatability import run

base_dir = '/HDD/etc/repeatablility'
# dir_names = ['gcnet_epochs100', 'mask2former_epochs140', 'pidnet_l_epochs300', 'sam2_epochs300']
# filename_postfix = '_3_0'
dir_names = ['2nd/deeplabv3plus_2']
filename_postfix = ''
rect_iou = True 
offset = 100

### ===================================
iou_thresholds = [0.05, 0.1, 0.2, 0.3]
area_thresholds = [20, 100, 150, 200]
figs = True 
vis = False
run(base_dir, dir_names, iou_thresholds, area_thresholds, vis, figs, rect_iou=rect_iou, 
    offset=offset, filename_postfix=filename_postfix)
### ===================================
iou_thresholds = [0.05]
area_thresholds = [20]
figs = False
vis = True
run(base_dir, dir_names, iou_thresholds, area_thresholds, vis, figs, rect_iou=rect_iou, 
    offset=offset, filename_postfix=filename_postfix)
                

# base_dir = '/HDD/etc/repeatablility'
# # dir_names = ['gcnet_epochs100', 'mask2former_epochs140', 'pidnet_l_epochs300', 'sam2_epochs300']
# # filename_postfix = '_3_0'
# dir_names = ['2nd/deeplabv3plus_2']
# filename_postfix = ''
# rect_iou = False
# offset = 0

# ### ===================================
# iou_thresholds = [0.05]
# area_thresholds = [20]
# figs = False
# vis = False
# run(base_dir, dir_names, iou_thresholds, area_thresholds, vis, figs, rect_iou=rect_iou, 
#     offset=offset, filename_postfix=filename_postfix)
                

        
    
        
    
    
    