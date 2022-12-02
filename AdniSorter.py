import csv
import math
import os
import random
import re
import shutil
import pydicom
import string
from matplotlib import pyplot as plt


class AdniSorter:

    def __init__(self):

        # The final result of the AdniSorter.run() operation
        self.cdrToImage = {}

        # Stores the index of the CDR and session ID in the CSV rows
        self.cdrIndex = 3
        self.sessionIdIndex = 5

        # Stores data from the CSV
        self.data = self.process_csv(os.path.join("ADNI", "T2 Axial - 6826 Samps CDR.csv"))


        topPath = os.path.join(os.getcwd(), "ADNI")
        sortedPath = os.path.join(topPath, "Sorted")

        # Stores base paths for the unsorted folder, training folders, and validation folders
        self.unsortedPath = os.path.join(topPath, "Unsorted")
        self.trainingPath = os.path.join(sortedPath, "train")
        self.validationPath = os.path.join(sortedPath, "validation")

        # Maps session IDs to their CDRs
        self.sessionToCDR = {}
        # Maps session IDs to their folder paths
        self.sessionToPath = {}
        # map session IDs to their desired images
        self.sessionToImages = {}

    def run(self):
        # Maps session IDs to their CDRs
        self.sessionToCDR = self.getSessionCDRs()

        # Maps session IDs to their folder paths
        self.sessionToPath = self.getSessionPaths()

        # Next, split each dicom image into a folder of images
        self.splitSessionDicoms()
        # map session IDs to their images
        # for session in self.sessionToPath:
        #     path = self.sessionToPath[session]
        #     self.sessionToImages[session] = self.getImagePaths(path)



        # Finally, map cdrs to lists of images
        self.getFinalSets()


        print("Processing complete.")
        nonLen = len(self.cdrToImage["NonDemented"])
        veryMildLen = len(self.cdrToImage["VeryMildDemented"])
        mildLen = len(self.cdrToImage["MildDemented"])
        modLen = len(self.cdrToImage["ModerateDemented"])
        
        final_count = nonLen + veryMildLen + mildLen + modLen
        
        print(f"Sort Results:\n Total Images: {final_count} \nNon Demented: {nonLen} "
              f"\nVery Mild Demented: {veryMildLen} \nMild Demented: {mildLen}"
              f"\nModerate Demented: {modLen}")

        # return self.cdrToImage



    def process_csv(self, filename_in):
        with open(filename_in, newline='') as file:
            reader = csv.reader(file, delimiter=',', quotechar='|')

            labels = []
            entries = []

            # Process the rows into lists and remove extra quotations
            rows = []
            for row in reader:
                row_list = [entry.strip('\"') for entry in row]

                # Don't append rows with empty CDRs or that are field mapping
                if row_list[3] != '' and row_list[4].lower() != 'field mapping':
                    rows.append(row_list)

            # Set the labels to the first row and strip it from the entries
            labels = rows[0]

            del rows[0]

            return rows

    # Goes through the CSV data and maps session IDs to their CDRs.
    def getSessionCDRs(self):
        sessionToCDR = {}

        for row in self.data:
            cdr = row[self.cdrIndex]
            sessionId = row[self.sessionIdIndex]

            if (cdr != '' and cdr != None):
                sessionToCDR[sessionId] = cdr

        print(f"Found {len(sessionToCDR)} session CDRs.")
        return sessionToCDR

    # Walks through every folder in ADNI/unsorted, finds any folder with a name equivalent to a session ID,
    # and associates it with its filepath.
    def getSessionPaths(self):
        sessionToPath = {}
        # Regex for session IDs
        sesReg = re.compile(r"I\d{4,7}$")

        for folder, subfolders, files in os.walk(os.path.join("ADNI", "unsorted")):

            # Only searches the last ten digits of the current path
            match = sesReg.search(folder[(len(folder) - 10):])
            if match is not None and match.group is not None:

                # Checks to see if this session has been assigned a CDR
                if match.group(0)[1:] in self.sessionToCDR:
                    sessionToPath[match.group(0)[1:]] = folder

        print(f"Found {len(sessionToPath)} session folders.")
        return sessionToPath

    # Splits each dicom in each session into its respective images
    def splitSessionDicoms(self):
        median_n = 5
        dcmList = []
        i = 0
        for session in self.sessionToPath:
            path = self.sessionToPath[session] # session folders
            slices = pydicom.data.data_manager.get_files(path, '*')
            if len(slices)==1:
                filename = slices[0]
                medianSlices = self.handleSingleSlice(median_n, filename)

                for num, pixel_array in enumerate(medianSlices):
                    new_file_name = filename.replace('.dcm','.jpg').replace('_1_',str(num))
                    plt.imsave(new_file_name, pixel_array, cmap=plt.cm.bone)
                    if session not in self.sessionToImages:
                        self.sessionToImages[session] = [new_file_name]
                    else:
                        self.sessionToImages[session].append(new_file_name)
            else:
                medianSlices = self.getNMedianSlices(median_n,slices)

                for slice in medianSlices:
                    pixel_array = pydicom.dcmread(slice).pixel_array
                    new_file_name = slice.replace('.dcm','.jpg')
                    plt.imsave(new_file_name, pixel_array, cmap=plt.cm.bone)
                    if session not in self.sessionToImages:
                        self.sessionToImages[session] = [new_file_name]
                    else:
                        self.sessionToImages[session].append(new_file_name)

    
    # Takes the filenames of all the slices in a single session folder and 
    # returns the ones that are the median N
    def getNMedianSlices(self, N, slices):
        medianSlices = [] # stores median n slices (filenames)
        medianSliceNumbers = [] # stores median n numbers
        if(len(slices)==1): raise Exception("Error")
        regSlice = re.compile('_\d{1,3}_')
        sliceNumbers = []
        for slice in slices:
            matches = regSlice.findall(slice)
            if matches is not None and len(matches)>0:
                numString = matches[-1]
                num = int(numString[1:-1])
                sliceNumbers.append(num)

        firstSliceNumber = min(sliceNumbers)
        lastSliceNumber = max(sliceNumbers)
        medianSliceNumber = ((lastSliceNumber-firstSliceNumber)//2) + firstSliceNumber
        if medianSliceNumber%2 != 0: medianSliceNumber += 1 
        halfN = N//2
        
        medianSliceNumbers.append(medianSliceNumber)
        for n in range(halfN):
            medianSliceNumbers.append(medianSliceNumber+2*(n+1))
            medianSliceNumbers.append(medianSliceNumber-2*(n+1))

        ## Convert the numbers to the format you'd find them in the filenames
        numberStrings = []
        for number in medianSliceNumbers:
            numberStrings.append("_{}_".format(number))

        ## If the filename contains the number, add it to medianSlices
        for slice in slices:
            if any(number in slice[-30:] for number in numberStrings):
                medianSlices.append(slice)

        return medianSlices

    # If the session only consists of a single dicom, pass in the filename and
    # return the median n pixel arrays
    def handleSingleSlice(self,N,filename):
        file = pydicom.dcmread(filename)
        pixel_array_3d = file.pixel_array
        num_slices, dimX, dimY = pixel_array_3d.shape

        medianSliceArrays = []
        halfN = N//2

        medianSliceIndex = num_slices//2
        if medianSliceIndex%2 != 0: medianSliceIndex+=1
        medianSliceArrays.append(pixel_array_3d[medianSliceIndex])
        for n in range(halfN):
            medianSliceArrays.append(pixel_array_3d[medianSliceIndex+(n+1)])
            medianSliceArrays.append(pixel_array_3d[medianSliceIndex-(n+1)])

        return medianSliceArrays

    # Uses the final sessionToCDR and sessionToImage maps,
    # and compiles images into four lists by alzheimers rating
    def getFinalSets(self):

        self.cdrToImage["NonDemented"] = list()
        self.cdrToImage["VeryMildDemented"] = list()
        self.cdrToImage["MildDemented"] = list()
        self.cdrToImage["ModerateDemented"] = list()

        for session in self.sessionToCDR:
            cdr = self.sessionToCDR[session]
            imagePaths = self.sessionToImages[session]

            match cdr:
                case 0.0:
                    self.cdrToImage["NonDemented"].extend(imagePaths)
                case 0.5:
                    self.cdrToImage["VeryMildDemented"].extend(imagePaths)
                case 1.0:
                    self.cdrToImage["MildDemented"].extend(imagePaths)
                case 2.0:
                    self.cdrToImage["ModerateDemented"].extend(imagePaths)
                case 3.0:  # Ignore these, there are too few available.
                    pass
                case _:  # Something has gone wrong if we get here
                    raise Exception(f"getFinalSets: CDR {cdr} is invalid.")

    # Takes a dictionary of (alzheimers rating, image paths) and a validation fraction.
    # Moves that percentage of images to self.validationPath
    # Moves the remainder to self.trainingPath
    # Validate fraction: Should be a number between 0 and 1. Ideally 0.1 or 0.2
    def sortValidate(self, validate_fraction):

        # Get image paths of each type
        nonDem = self.cdrToImage["NonDemented"]
        veryMildDem = self.cdrToImage["VeryMildDemented"]
        mildDem = self.cdrToImage["MildDemented"]
        modDem = self.cdrToImage["ModerateDemented"]

        # Go through each type, get the correct validation fraction, and move them to the validation folder

        ##### Non dem sample
        length = len(nonDem)
        random.shuffle(nonDem)
        validationSet = nonDem[:((length + 1) * validate_fraction)]  # Splits validation fraction
        trainingSet = nonDem[((length + 1) * validate_fraction):]  # Split remainder to test set

        for item in validationSet:
            shutil.move(item, os.path.join(self.validationPath, "NonDemented"))
        for item in trainingSet:
            shutil.move(item, os.path.join(self.trainingPath, "NonDemented"))

        #### Very mild dem sample
        length = len(veryMildDem)
        random.shuffle(veryMildDem)
        validationSet = veryMildDem[:((length + 1) * validate_fraction)]  # Splits validation fraction
        trainingSet = veryMildDem[((length + 1) * validate_fraction):]  # Split remainder to test set

        for item in validationSet:
            shutil.move(item, os.path.join(self.validationPath, "VeryMildDemented"))
        for item in trainingSet:
            shutil.move(item, os.path.join(self.trainingPath, "VeryMildDemented"))

        # Mild Dem sample
        length = len(mildDem)
        random.shuffle(mildDem)
        validationSet = mildDem[:((length + 1) * validate_fraction)]  # Splits validation fraction
        trainingSet = mildDem[((length + 1) * validate_fraction):]  # Split remainder to test set

        for item in validationSet:
            shutil.move(item, os.path.join(self.validationPath, "MildDemented"))
        for item in trainingSet:
            shutil.move(item, os.path.join(self.trainingPath, "MildDemented"))

        # Mod dem sample
        length = len(modDem)
        random.shuffle(modDem)
        validationSet = modDem[:((length + 1) * validate_fraction)]  # Splits validation fraction
        trainingSet = modDem[((length + 1) * validate_fraction):]  # Split remainder to test set

        for item in validationSet:
            shutil.move(item, os.path.join(self.validationPath, "ModerateDemented"))
        for item in trainingSet:
            shutil.move(item, os.path.join(self.trainingPath, "ModerateDemented"))

if __name__ == "__main__":
    sorter = AdniSorter()
    sorter.run()

    print(len(sorter.cdrToImage["NonDemented"]))
    print(len(sorter.cdrToImage["VeryMildDemented"]))
    print(len(sorter.cdrToImage["MildDemented"]))
    print(len(sorter.cdrToImage["ModerateDemented"]))

    # Test sorting into validation and train
    sorter.sortValidate(0.1)