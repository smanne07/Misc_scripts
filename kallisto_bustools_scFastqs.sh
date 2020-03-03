#!/bin/bash
set -e

if [ -f /etc/profile.d/modules.sh ]; then
   source /etc/profile.d/modules.sh
fi

#script which takes scRNAseq fastq files as input and runs kallisto bustools
##script expects folders samplewise with fastq Ref_files

##Use wherrylab modules and load the kallisto and bustools privatemodules

module use.own /project/wherrylab/wlabtools/privatemodules/
module load kallisto-0.46.1
module load bustools-0.39.3

base_dir=$1

#Move to the directory with the raw files

cd $base_dir

for dir in `ls -d */`;
do
filename="${dir%*/*}"
echo ${dir}
echo ${filename}

cd ${base_dir}/${dir}

for filename in *_1.fastq;
do echo $filename
R1=$filename;
R2=$(echo $R1 | sed 's/_R1/_R2/g')


kallisto bus -i /project/wherrylab/wlabtools/Ref_files/kallisto_bustools/mus_musculus/transcriptome.idx -o kallisto_bus_out -x 10xv5 -t8 $R1 $R2

bustools correct -w /project/wherrylab/wlabtools/Ref_files/kallisto_bustools/737K-august-2016.txt -o output.correct.bus output.bus

bustools sort -t 4 -o output.correct.sort.bus output.correct.bus

mkdir eqcount
mkdir genecount
bustools count -o eqcount/tcc -g /project/wherrylab/wlabtools/Ref_files/kallisto_bustools/transcripts_to_genes.txt -e matrix.ec -t transcripts.txt output.correct.sort.bus
bustools count -o genecount/gene -g /project/wherrylab/wlabtools/Ref_files/kallisto_bustools/transcripts_to_genes.txt -e matrix.ec -t transcripts.txt --genecounts output.correct.sort.bus

done
done
