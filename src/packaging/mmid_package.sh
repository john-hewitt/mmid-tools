# Take S3 image packages, emit the following:
#       - Language image package
#       - Mini language image package
#       - Language Metadata file

#names=( afrikaans-package.tar albanian-package.tar amharic-package.tar arabic-package.tar aragonese-package.tar armenian-package.tar asturian-package.tar azerbaijani-package.tar basque-package.tar belarusian-package.tar bengali-package.tar bishnupriya-manipuri-package.tar bosnian-package.tar breton-package.tar bulgarian-package.tar catalan-package.tar cebuano-package.tar central-bicolano-package.tar chinese-package.tar croatian-package.tar czech-package.tar danish-package.tar dutch-package.tar english-01-package.tar english-02-package.tar english-03-package.tar english-04-package.tar english-05-package.tar english-06-package.tar english-07-package.tar english-08-package.tar english-09-package.tar english-10-package.tar english-11-package.tar english-12-package.tar english-13-package.tar english-14-package.tar english-15-package.tar english-16-package.tar english-17-package.tar english-18-package.tar english-19-package.tar english-20-package.tar english-21-package.tar english-22-package.tar english-23-package.tar english-24-package.tar english-25-package.tar english-26-package.tar english-27-package.tar esperanto-package.tar filipino-package.tar finnish-package.tar french-package.tar frisian-package.tar galician-package.tar georgian-package.tar german-package.tar greek-package.tar gujarati-package.tar haitian-package.tar hebrew-package.tar hindi-package.tar hungarian-package.tar icelandic-package.tar ido-package.tar ilokano-package.tar indonesian-package.tar irish-package.tar italian-package.tar japanese-package.tar javanese-package.tar kannada-package.tar kapampangan-package.tar kazakh-package.tar korean-package.tar kurdish-package.tar latvian-package.tar lithuanian-package.tar low-saxon-package.tar luxembourgish-package.tar macedonian-package.tar malagasy-package.tar malay-package.tar malayalam-package.tar marathi-package.tar neapolitan-package.tar nepali-package.tar newar-package.tar norwegian-nynorsk-package.tar norwegian-package.tar pashto-package.tar persian-package.tar piedmontese-package.tar polish-package.tar portuguese-package.tar punjabi-package.tar romanian-package.tar russian-package.tar serbian-package.tar serbo-croatian-package.tar sicilian-package.tar sindhi-package.tar slovak-package.tar slovenian-package.tar somali-package.tar spanish-package.tar sundanese-package.tar swahili-package.tar swedish-package.tar tamil-package.tar telugu-package.tar thai-package.tar turkish-august-package.tar turkish-package.tar uighur-package.tar ukrainian-package.tar urdu-package.tar uzbek-package.tar vietnamese-package.tar waray-package.tar welsh-package.tar wolof-package.tar yoruba-package.tar )
names=( danish-package.tar )

for package_name in "${names[@]}"
do 
    echo $package_name
    # Pull down package from S3 and untar, putting it in a folder
    aws s3 cp s3://brendan.callahan.thesis/packages/$package_name .
    old_package_name=`echo $package_name | sed 's/.tar//'`
    mkdir $old_package_name/
    tar xvf $package_name -C $old_package_name/
    new_package_name=scale-`echo $package_name | sed 's/.tar//'`

    # Scale images and put them in a new folder
    mkdir $new_package_name
    bash scale_package.sh $old_package_name $new_package_name

    # Tar up the new package!
    tar czvf ${new_package_name}.tgz $new_package_name/

    # Make the mini-package!
    mini_package_name=mini-`echo $package_name | sed 's/.tar//'`
    mkdir $mini_package_name
    bash make_mini_package.sh $new_package_name $mini_package_name

    # Make the metadata file!
    metadata_name=metadata-`echo $package_name | sed 's/.tar//'`
    python package_metadata.py $new_package_name $metadata_name

done

