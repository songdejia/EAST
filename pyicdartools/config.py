#!/usr/bin/env python2
#encoding: UTF-8
import json
#Name of the script used for the evalution
evaluation_script = 'TL_iou'
#Custom evalution params
evaluation_params = json.loads("""{"IOU_CONSTRAINT":0.5,"AREA_PRECISION_CONSTRAINT":0.5,"GT_SAMPLE_NAME_2_ID":"gt_img_([0-9]+).txt","DET_SAMPLE_NAME_2_ID":"res_img_([0-9]+).txt","LTRB":false,"CRLF":false,"CONFIDENCES":false}""")
#Upload instructions
instructions = """<ul>
	<li>A single zip file is expected, containing a set of text files.</li>
	<li>No directory structure within the zip file is permitted, just the set of text files.</li>
	<li>The containing text files should be named as&nbsp;<strong><em>res_img_#.txt</em></strong>, where&nbsp;<em><strong>#</strong></em>&nbsp;is the number of the corresponding test-set image.</li>
	<li>Each text file should contain as many lines as text bounding boxes found. Each line should contain eight comma separated values only. The values should correspond to the coordinates of the four corners of the bounding quadrilateral of the word.</li>
	<li>New lines in the text files should be indicated with the windows CR/LF termination.</li>
</ul>

<p>The submitted zip file is automatically checked at the time of submission, and a submission log is presented to the user along with a confirmation of the submission. The checks performed are the following:</p>

<ul>
	<li>That the file submitted is a valid zip file, it can be opened and the contents can be extracted.</li>
	<li>That the names of the text files contained are correct and the image numbers are within the bounds of the test set.</li>
	<li>That each text file contains eight comma separated values per line.</li>
	<li>That the coordinates passed are within the bounds of the image and that the coordinates are in clocwise order</li>
</ul>

<p>See here an example of the&nbsp;<a href="http://rrc.cvc.uab.es/files/task1_ch4_sample.zip">expected submission file</a></p>
"""
#Extension of the GT file. gt.[extension]
gt_ext = "zip"
#Acronym for the task. It's used to cache the Images
acronym = "IST-T1"
#Title of the Task
title = "Incidental Scene Text - Task 1 Text Localization TEST DATASET (evaluation:IoU)"
#Custom JavaScript for the visualiztion.
customJS = 'visualization_TL_iou.js'
#Custom CSS for the visualiztion.
customCSS = 'visualization_TL_iou.css'
#Parameters used to show the results of a method and the method's ranking
method_params = json.loads("""{"recall":{"long_name":"Recall","type":"double","order":"","grafic":"1","format":"perc"},"precision":{"long_name":"Precision","type":"double","order":"","grafic":"1","format":"perc"},"hmean":{"long_name":"Hmean","type":"double","order":"desc","grafic":"1","format":"perc"}}""")
#Parameters to show for each sample
sample_params = json.loads("""{"recall":{"long_name":"Recall","type":"double","order":"","grafic":"","format":"perc"},"precision":{"long_name":"Precision","type":"double","order":"","grafic":"","format":"perc"},"hmean":{"long_name":"Hmean","type":"double","order":"desc","grafic":"","format":"perc"}}""")
#Parameters to ask for for each submition
submit_params = json.loads("""{}""")
#Regular expression to get the Sample ID from the image name. ID must be the first capturing group.
image_name_to_id_str = 'img_([0-9]+).(jpg|gif|png)'
