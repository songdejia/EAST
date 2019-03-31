#data
dataroot = './dataset'  #./dataroot/train/img      ./dataroot/train/gt
test_img_path = './data/test/img'
result = './result'

lr = 0.0001
gpu_ids = [0]
gpu = 1
init_type = 'xavier'

resume = False
checkpoint = ''# should be file
train_batch_size_per_gpu = 14
num_workers = 0  #bug2

print_freq = 1
eval_iteration = 500
save_iteration = 500
max_epochs = 1000







