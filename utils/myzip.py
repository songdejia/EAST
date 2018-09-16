import os
def MyZip(out_txt_dir, epoch):
	assert os.path.isdir(out_txt_dir), 'Res_txt dir is not exit'

	print('EAST <==> Evaluation <==> Into out_txt_dir:{} <==> Begin'.format(out_txt_dir))
	try:
		os.chdir(os.path.join(out_txt_dir, '{}'.format(epoch)))

		os.system('zip -r submit-{}.zip ./*.txt'.format(epoch))

		os.system('cp submit={}.zip ../../submit.zip'.format(epoch))

		os.chdir('../../')

		workspace = os.path.abspath('./')
	
	except:
		sys.exit('ZIP ERROR')

	print('EAST <==> Evaluation <==> Into out_txt_dir:{} <==> Done'.format(out_txt_dir))

	submit_path = os.path.join(workspace, 'submit.zip')

	return submit_path







