
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.functional.renderer.OWLFunctionalSyntaxRenderer
import org.semanticweb.owlapi.io.AbstractOWLRenderer
import org.semanticweb.owlapi.io.OWLRendererException
import org.semanticweb.owlapi.krss2.renderer.KRSS2OWLSyntaxRenderer
import org.semanticweb.owlapi.krss2.renderer.KRSS2SyntaxRenderer
import org.semanticweb.owlapi.krss2.renderer.KRSSSyntaxRenderer
import org.semanticweb.owlapi.latex.renderer.LatexRenderer
import org.semanticweb.owlapi.manchestersyntax.renderer.ManchesterOWLSyntaxRenderer
import org.semanticweb.owlapi.model.IRI
import org.semanticweb.owlapi.model.OWLDataFactory
import org.semanticweb.owlapi.model.OWLOntology
import org.semanticweb.owlapi.model.OWLOntologyCreationException
import org.semanticweb.owlapi.model.OWLOntologyManager
import org.semanticweb.owlapi.owlxml.renderer.OWLXMLRenderer
import org.semanticweb.owlapi.reasoner.InferenceType

import de.tudresden.inf.lat.jcel.owlapi.main.*
import de.tudresden.inf.lat.jcel.reasoner.main.VersionInfo
import de.tudresden.inf.lat.jcel.reasoner.main.*

import de.tudresden.inf.lat.jcel.ontology.normalization.*
import de.tudresden.inf.lat.jcel.ontology.axiom.extension.*
import de.tudresden.inf.lat.jcel.coreontology.axiom.*
import de.tudresden.inf.lat.jcel.owlapi.translator.*
import de.tudresden.inf.lat.jcel.ontology.axiom.complex.*

// input: OWL file
File ontologyFile = new File(args[0])
// output: contains all normalized axioms in OWL Functional Syntax
PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter(args[1])))

OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
OWLOntology ontology = manager.loadOntologyFromOntologyDocument(ontologyFile)
JcelReasoner ret = new JcelReasoner(ontology, false)
def translator = ret.getTranslator()
Set<ComplexIntegerAxiom> iOntology = ret.getIntegerOntology()
OntologyNormalizer normalizer = new OntologyNormalizer()
IntegerOntologyObjectFactory factory = new IntegerOntologyObjectFactoryImpl()
Set<NormalizedIntegerAxiom> normalizedOntology = normalizer.normalize(iOntology, factory)

ReverseAxiomTranslator rTranslator = new ReverseAxiomTranslator(translator, ontology)

normalizedOntology.each { ax ->
    try {
	fout.println(rTranslator.visit(ax))
    } catch (Exception E) {
	println "Ignoring $ax:" + E.getMessage()
    }
}

fout.flush()
fout.close()

