#just a quick script for downloading required dataasets


datasets=[]
#fill this in later

echo "starting dataset download"
for dataset in datasets do
	echo "downloading $dataset"
	mkdir $dataset
	cd $dataset
	#do the download - for now
	wget $dataset
	#finish
	echo "finished downloading $dataset"
	cd ..
	fi
echo "completed download of datasets"
