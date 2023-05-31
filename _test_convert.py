# Importer la librairie
#import convertapi
import os
import PyPDF2
import random




def convert2txt(file_name):
    base_name, extension = os.path.splitext(file_name)
    if (extension==".pdf"):
        #convertir en txt
        # Ouvrir le fichier pdf en mode lecture binaire
        pdf_file = open(file_name, 'rb')

        # Créer un objet PdfFileReader
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)

        # Ouvrir le fichier txt en mode écriture
        output_filename = file_name + str(random.rand(1,10000)) + ".txt"
        txt_file = open(output_filename, 'w')

        # Parcourir toutes les pages du pdf
        for page in pdf_reader.pages:
            # Extraire le texte de la page
            text = page.extractText()
            # Écrire le texte dans le fichier txt
            txt_file.write(text)

        # Fermer les fichiers
        pdf_file.close()
        txt_file.close()
        return(output_filename)
    else :
        if (extension==".txt"):
            return(file_name)
        else :
            print("Le fichier n'est pas un pdf ou un txt")
            return("")
