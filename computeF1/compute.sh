

#python compute f1
cd ./computeF1
echo "in computeF1"

cp ../submit.zip ./

python2 script.py –g=gt.zip –s=submit.zip –o=./
echo "exc script.py"

cd ../
echo "go out"


