import os
import sys
def MyZip(out_txt_dir, epoch):
	print('ZIP :get out_txt_dir {}'.format(out_txt_dir))
	#out_txt_dir : /home/djsong/update/result/epoch_0_gt
	assert os.path.isdir(out_txt_dir), 'Res_txt dir is not exit'

	print('EAST <==> Evaluation <==> Into out_txt_dir:{} <==> Begin'.format(out_txt_dir))
	try:
		os.chdir(out_txt_dir)

		os.system('zip -r submit-{}.zip ./*.txt'.format(epoch))

		os.system('cp submit-{}.zip ../submit.zip'.format(epoch))

		os.chdir('../../')

	
	except:
		sys.exit('ZIP ERROR')

	print('EAST <==> Evaluation <==> Into out_txt_dir:{} <==> Done'.format(out_txt_dir))
	workspace = os.path.abspath('./result')

	submit_path = os.path.join(workspace, 'submit.zip')

	return submit_path







