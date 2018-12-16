#!/bin/sh
#LANG=Albanian
#INPUT_DIR="/scratch-shared/users/bcal/packages/$LANG"
#OUTPUT_DIR="/nlp/users/johnhew/tmp"
INPUT_DIR=$1
OUTPUT_DIR=$2
cd $INPUT_DIR
mkdir -p $OUTPUT_DIR
INDEX_FILE="$OUTPUT_DIR/index.tsv"
if [[ -f "$INDEX_FILE" ]]
then
  echo 'Warning: Index file already exists. Starting again from scratch?'
  read -r -p "[Y/n]: " response
  response=${response,,} # tolower
  if [[ $response =~ ^(yes|y| ) ]] || [[ -z $response ]]
  then
    echo 'Continuing'
  else
    echo 'Exiting'
    exit 1
  fi
  rm $INDEX_FILE 
fi
  
echo '====== Compressining images ======'
for WORD_DIR in *
do
  if [[ -d $WORD_DIR ]];
  then
    OUTPUT_WORD_DIR="$OUTPUT_DIR/$WORD_DIR"
    echo "Processing $WORD_DIR"
    echo "Saving to $OUTPUT_WORD_DIR"
    # TODO(daphne): Do we want to skip instead?
    if [[ -d $OUTPUT_WORD_DIR ]]
    then
      echo 'Warning: output directory already exists, deleting it.'
      rm -rf $OUTPUT_WORD_DIR
    fi
    mkdir "$OUTPUT_WORD_DIR"
    time mogrify -background white -alpha off -resize "256>"   -format jpg -path "${OUTPUT_WORD_DIR}" "${WORD_DIR}/*.*[0]"
    
    # Remove random extra jpg that is getting made.
    rm "$OUTPUT_WORD_DIR/word.jpg"
    
    # Copy over the metadata
    cp $WORD_DIR/*.json $OUTPUT_WORD_DIR
    cp $WORD_DIR/word.txt $OUTPUT_WORD_DIR
    # Write word and path name to index file.
    word=$(cat "$OUTPUT_WORD_DIR/word.txt")
    echo -e "${word}\t${WORD_DIR}" >> $INDEX_FILE
    echo 'Done'
    echo ''
  fi
done
echo "Finished porocessing: ${LANG}"
