# if you don't have PyPDF2: conda install -c conda-forge pypdf2
from PyPDF2 import PdfFileReader, PdfFileWriter  
from os import replace
import tempfile


def add_pdf_metadata(pdf_file, metadata_dict):
    """
    Adds a Python dict to the metadata in a pdf_file.
    Uses PyPDF2 and a temporary file.
    There are no doubt better ways to do this! 
    PyPDF2 needs the keys in the dict to have a "/" in front; these are added
     automagically by this function. 
    
    Parameters
    ----------
    pdf_file : str
        The pdf file to be modified.
    metadata_dict : dict
        dict to be added to the pdf metadata (without "/" in front of names,
          no spaces in names, and only strings for keys).
    """
    
    assert isinstance(metadata_dict, dict), \
                      "2nd argument must be a dict"
    
    fin = open(pdf_file, 'rb')
    reader = PdfFileReader(fin)
    
    writer = PdfFileWriter()
    writer.appendPagesFromReader(reader)
            
        
    metadata = reader.getDocumentInfo()
    writer.addMetadata(metadata)
    
    # Append the custom metadata dict
    metadata_dict_added = {}
    for key, value in metadata_dict.items():
        new_key = "/" + key
        metadata_dict_added[new_key] = value

    writer.addMetadata(metadata_dict_added)
    
    # Write out the full pdf file to a temporary file
    _, temp_file_path = tempfile.mkstemp()
    fout = open(temp_file_path, 'wb')
    writer.write(fout)
    
    # Close up the input and output streams 
    fin.close()
    fout.close()
    
    # Move the temporary file to the original file
    replace(temp_file_path, pdf_file)


def get_pdf_metadata(pdf_file, exclude=True):
    """
    Get the metadata from a pdf as a dict, stripping the leading "/"s.
    Uses PyPDF2.
    
    Parameters
    ----------
    pdf_file : str
        The pdf file to get the metadata from.
    exclude : bool, optional
        Only return the custom metadata. Default: True.
    """

    fin = open(pdf_file, 'rb')
    reader = PdfFileReader(fin)
    metadata = reader.getDocumentInfo()
    
    # Close up the input stream 
    fin.close()
    
    if exclude:
        pdf_keys = ["/Producer", "/CreationDate", "/Creator", "/Author",
                    "/Subject", "/Title", "/Keywords", "/ModDate"]
        for key in pdf_keys:
            if key in metadata:
                del metadata[key]

    stripped_metadata = {}
    for key, value in metadata.items():
        new_key = key.replace("/", "", 1)
        stripped_metadata[new_key] = value
                
    return stripped_metadata
