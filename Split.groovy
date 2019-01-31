def cli = new CliBuilder()
cli.with {
usage: 'Self'
  h longOpt:'help', 'this information'
  i longOpt:'input', 'input STRING file', args:1, required:true
  o longOpt:'output', 'output file containing generated ontology',args:1, required:true
  t longOpt:'test-output', 'output file containing generated testing data',args:1, required:true
}
def opt = cli.parse(args)
if( !opt ) {
  //  cli.usage()
  return
}
if( opt.h ) {
    cli.usage()
    return
}

PrintWriter fout1 = new PrintWriter(new FileWriter(opt.o))
PrintWriter fout2 = new PrintWriter(new FileWriter(opt.t))
Set<String> set = new HashSet<String>()

new File(opt.i).splitEachLine("\t") { line ->
    if (!line[0].startsWith("item")) {
	def id1 = line[0]
	def id2 = line[1]
	def rel = line[2]
	def score = new Integer(line[-1])
	if (score >= 700) {  // only use high-confidence predictions
	    set.add("$id1\t$id2\t$rel")
	}
    }
}
set.each {
    if (Math.random() > 0.1) {
	fout1.println("$it")
    } else {
	fout2.println("$it")
    }
}
fout1.flush()
fout2.flush()
fout1.close()
fout2.close()
