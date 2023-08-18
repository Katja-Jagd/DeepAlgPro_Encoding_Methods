# DeepAlgPro_Encoding_Methods
Extension of the DeepAlgPro software with different encoding methods

See https://github.com/chun-he-316/DeepAlgPro/blob/main/README.md for details.

Two additional command line arguments have been added to the main.py script and the predict.py

Main.py

-h, --help                     show this help message and exit

-i INPUTS, --inputs INPUTS     input file

--epochs N                     number of total epochs to run

--lr LR, --learning-rate LR    learning rate

-b N, --batch-size N

--mode {train,test}

-et,--encoding-type            Encoding type, options One_hot, One_hot_6_bit, Binary_5_bit, Hydrophobicity matrix, Meiler_parameters, 
                               Acthely factors, PAM250, BLOSUM62, Miyazawa energies, Micheletti potentials, AESNN3, ANN4D, ProtVec, ESM2
                               
-ed, --encoding-directory      Directory path to the encoding data 

Predict.py

-h, --help                    show this help message and exit

-i INPUTS, --inputs INPUTS    input file

-b N, --batch-size N

-o OUTPUT, --output OUTPUT    output file

-et,--encoding-type            Encoding type, options One_hot, One_hot_6_bit, Binary_5_bit, Hydrophobicity matrix, Meiler_parameters,  
                               Acthely factors, PAM250, BLOSUM62, Miyazawa energies, Micheletti potentials, AESNN3, ANN4D, ProtVec, ESM2
                               
-ed, --encoding-directory      Directory path to the encoding data 

The files necessary to run the ESM-2 encoding is not uploaded to this project yet. If needed use the "Compute embeddings in bulk from FASTA" from https://github.com/facebookresearch/esm to produce the files yourself. 
