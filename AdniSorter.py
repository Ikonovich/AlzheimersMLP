import csv
import math
import os
import random
import re
import shutil


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

        # map session IDs to their images
        for session in self.sessionToPath:
            path = self.sessionToPath[session]
            self.sessionToImages[session] = self.getImagePaths(path)

        # Finally, map cdrs to lists of images
        self.getFinalSets()

        nonLen = len(self.cdrToImage["NonDemented"])
        veryMildLen = len(self.cdrToImage["VeryMildDemented"])
        mildLen = len(self.cdrToImage["MildDemented"])
        modLen = len(self.cdrToImage["ModerateDemented"])

        final_count = nonLen + veryMildLen + mildLen + modLen


        print(f"Sort Results:\n Total Images: {final_count} \nNon Demented: {nonLen} "
              f"\nVery Mild Demented: {veryMildLen} \nMild Demented: {mildLen}"
              f"\nModerate Demented: {modLen}")

        return self.cdrToImage



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
        sesReg = re.compile(r"I\d{5, 6}")
        for folder, subfolders, files in os.walk(os.path.join("ADNI", "unsorted")):

            if sesReg.match(folder):
                sessionToPath[folder] = os.path.join(self.unsortedPath, folder)

        print(f"Found {len(sessionToPath)} session folders.")
        return sessionToPath

    ## Gets the images we want for a specific session path
    def getImagePaths(self, sessionPath):

        # Stores the image paths we want to return
        selectedImages = []

        # Regex matcher for image number
        numMatch = re.compile(r"_\d{2, 3}_")

        # Stores all image names in the session folder
        imagePaths = os.listdir(sessionPath)

        # Get what should be the median picture index
        median = int(len(imagePaths) / 2)

        # Maps image numbers to their paths
        numToPath = {}
        for file in imagePaths:

            # Split the string to avoid false positives,
            # The file number is always late in the string
            splitPoint = int(len(file) / 2)
            name = file[splitPoint:]

            results = numMatch.search(name).group(0)
            results.replace("_", "")
            num = int(results)
            # Adds the image's number and its path to the dictionary
            numToPath[num] = os.path.join(sessionPath, file)

        return self.filterDesired(numToPath, median)

    # Takes a dictionary of (image number, image path) and returns the
    # 5 closest to the middle
    def filterDesired(self, numToPathDict, median):

        desiredImages = []
        for i in range(5):
            closestNum = 0

            for num in numToPathDict:
                if math.abs(median - num) < math.abs(median - closestNum):
                    closestNum = num
            desiredImages.append(numToPathDict[closestNum])
            del numToPathDict[closestNum]

        return desiredImages

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
    def sortValidate(self, cdrToImageMap, validate_fraction):

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
    sorter.sortValidate(sorter.cdrToImage, 0.1)
