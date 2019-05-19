org=$1
org_id=$2
#echo Generating split for ${org} ${org_id}
#groovy Split.groovy -i data/data/${org_id}.protein.links.v10.5.txt -o data/data-train/${org_id}.protein.links.v10.5.txt -v data/data-valid/${org_id}.protein.links.v10.5.txt -t data/data-test/${org_id}.protein.links.v10.5.txt
#echo Generating plain data for ${org} ${org_id}
#groovy GeneratePlainTrainingData.groovy -i data/data-train/${org_id}.protein.links.v10.5.txt -o data/data-train/${org}-plain.nt

#echo Generating owl for $org $org_id
#groovy GenerateTrainingDataClasses.groovy -i data/data-train/${org_id}.protein.links.v10.5.txt -o data/data-train/${org}-classes.owl

#echo Normalizing owl for $org $org_id
#groovy -cp jar/jcel.jar Normalizer.groovy -i data/data-train/${org}-classes.owl -o data/data-train/${org}-classes-normalized.owl

echo Generating rdf owl for $org $org_id
groovy GenerateTrainingDataEL.groovy -i data/data-train/${org_id}.protein.links.v10.5.txt -o data/data-train/${org}-classes-rdf.owl

echo Converting OWL to RDF/NT
rapper data/data-train/${org}-classes-rdf.owl -o ntriples > data/data-train/${org}-classes-rdf.nt
