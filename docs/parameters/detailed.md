CellRex Documentation

-----

# Upload
## **Upload**

### Definition
Selection of the file to be uploaded.
### Usage Guidelines
After uploading a file, the initial selection of a new file will not be adopted.
### Attributes
- Datatype: Enumeration, String, File
- Range: One of the files in the specified Upload Folder
- Optional: No
### Dependencies
- Configuration pointing to a file location
- List of assets in the file location

# Subject
## **Species**

### Definition
The specific biological species from which the research sample is derived, providing fundamental taxonomic classification.
### Examples
- Human
- Mouse
- Rat
- ....
### Attributes
- Datatype: Enumeration, String
- Range: One of predefined from list in configuration
- Optional: No
### Dependencies
- Configuration