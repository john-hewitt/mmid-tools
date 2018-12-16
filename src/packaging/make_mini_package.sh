INPUT_DIR=$1
OUTPUT_DIR=$2

for i in $( ls $INPUT_DIR ); 
do
    mkdir $OUTPUT_DIR/$i
    cp $INPUT_DIR/$i/word.txt $OUTPUT_DIR/$i/
    cp $INPUT_DIR/$i/metadata.json $OUTPUT_DIR/$i/
    cp $INPUT_DIR/$i/errors.json $OUTPUT_DIR/$i/
    cp $INPUT_DIR/$i/01.jpg $OUTPUT_DIR/$i/
done
    
