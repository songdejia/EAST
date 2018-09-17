#data
dataroot='./data'  #./data/train/img      ./data/train/gt
test_img_path='./data/test/img'
result = './result'

lr = 0.0001
gpu_ids = [0]
gpu = 1
init_type = 'xavier'

resume = False
checkpoint = ''# should be file
train_batch_size_per_gpu  = 14
num_workers = 1

print_freq = 1
eval_iteration = 50
save_iteration = 50
max_epochs = 1000000







