

params.transcriptomeindex = "/project/wherrylab/wlabtools/Ref_files/kallisto_bustools/mus_musculus/transcriptome.idx"
params.whitelist = "/project/wherrylab/wlabtools/Ref_files/kallisto_bustools/737K-august-2016.txt"
params.t2g = "/project/wherrylab/wlabtools/Ref_files/kallisto_bustools/mus_musculus/transcripts_to_genes.txt"
params.name          = "RNA-Seq Abundance Analysis"
params.output        = "KallistoBusresults/"
params.base_dir ="/home/sasim/RNAseq_data_analysis/PRJNA587711/rawfiles/fastq/"


/*
 * Input parameters validation
 */

transcriptomeindex     = file(params.transcriptomeindex)
whitelist     = file(params.whitelist)
t2g     = file(params.t2g)
base_dir = params.base_dir


/*
 * validate input files
 */
if( !transcriptomeindex.exists() ) exit 1, "Missing transcriptome file: ${transcriptomeindex}"

/*
 * Create a channel for read files
 */

Channel
    .fromFilePairs( params.reads, size: -1 )
    .ifEmpty { error "Cannot find any reads matching: ${params.reads}" }
    .set { read_files }



process mapping {
    tag "reads: $name"
    cpus 4
    memory 10.GB
    publishDir "${params.output}", mode: 'move'
    input:
    file index from transcriptomeindex
    file wlist from whitelist
    file tg from t2g

    set val(name), file(reads) from read_files

    output:
    file "kallisto_${name}" into kallisto_out_dirs

    script:
    //
    // Kallisto tools mapper
    //

        """
        mkdir kallisto_${name}
        kallisto bus -i ${index} -o kallisto_${name} -x 10Xv3  -t8 ${reads}
        cd kallisto_${name}
        bustools correct -w ../${wlist} -o output.correct.bus output.bus

        bustools sort -t 4 -o output.correct.sort.bus output.correct.bus

        mkdir eqcount
        mkdir genecount
        bustools count -o eqcount/tcc -g ../${tg} -e matrix.ec -t transcripts.txt output.correct.sort.bus
        bustools count -o genecount/gene -g ../${tg} -e matrix.ec -t transcripts.txt --genecounts output.correct.sort.bus
        cd ${base_dir}
        """

}
