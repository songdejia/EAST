#data
dataroot='./data'  #./data/train/img      ./data/train/gt
test_img_path='./data/test/img'
result = './result'

lr = 0.0001
gpu_ids = [0, 1]
gpu = 2
init_type = 'xavier'

resume = False
checkpoint = ''# should be file
train_batch_size  = 28
num_workers = 14

print_freq = 1
eval_iteration = 1000
save_iteration = 1000
max_epochs = 1000000







