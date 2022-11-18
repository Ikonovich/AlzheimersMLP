import csv
import statistics
import matplotlib.pyplot as plt
import pydicom
import pydicom.data
import os

def process_csv(filename_in):
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
        print(labels)
        print(rows[1])

        processed_rows = remove_excess(rows)
        analyze_data(processed_rows)


# Removes excess entries, keeping no more than 8 entries from any single subject.
def remove_excess(rows):
    # The number of allowed sessions before a subject is pruned.
    num_allowed = 8
    processed_rows = []

    # Stores the ID and session count of all subjects
    subject_session_count = {}
    # Stores the ID and sessions of all subjects
    subject_session_list = {}

    # Stores the ID of subjects determined to exceed the allowed number of imaging sessions.
    excess_subjects = []

    # Remove subjects with excess entries
    for row in rows:
        subject = row[0]
        if subject in subject_session_count:
            subject_session_count[subject] += 1
            subject_session_list[subject].append(row)
        else:
            subject_session_count[subject] = 1
            subject_session_list[subject] = [row]

    # for subject in subject_session_count:
    #     if subject_session_count[subject] > num_allowed:
    #         excess_subjects.append(subject)

    # Remove excess subjects
    processed_rows = []
    for row in rows:
        if row[0] not in excess_subjects:
            processed_rows.append(row)

    return processed_rows

# Finds the n most evenly distributed sessions in a list of sessions
# Returns the Ids of any *excluded* sessions
def find_mean_sessions(sessions, n):
    # Convert to a dictionary of sessionId : Age

    sess_age = {}
    for session in sessions:
        sess_age[session[5]] = session[2]

    return []






def analyze_data(rows):

    # Stores the number of each classification.
    no_dementia = 0
    very_mild_dementia = 0
    mild_dementia = 0
    moderate_dementia = 0
    severe_dementia = 0

    # Stores the ID and sex of all subjects
    subject_sex = {}
    # Stores the ID and session count of all subjects
    subject_sessions = {}
    # Stores the ID of all image sessions
    sessions = set()
    # Stores ages at time of scan
    ages = []
    # Counts the number of male, female, and unknown imaging sessions
    male_sessions = 0
    female_sessions = 0

    # Counts the number of male, female, and unknown subjects
    males = 0
    females = 0

    for row in rows:
        sessions.add(row[5])
        ages.append(float(row[2]))
        subject = row[0]
        subject_sex[subject] = row[1]

        if subject in subject_sessions:
            subject_sessions[subject] += 1
        else:
            subject_sessions[subject] = 1

        if row[1] == 'F':
            female_sessions += 1
        elif row[1] == 'M':
            male_sessions += 1

        cdr = float(row[3])

        if cdr < 0.5:
            no_dementia += 1
        elif cdr < 1:
            very_mild_dementia += 1
        elif cdr < 2:
            mild_dementia += 1
        elif cdr < 3:
            moderate_dementia += 1
        else:
            severe_dementia += 1

    # Count sexes of subjects
    for subject in subject_sex:
        if subject_sex[subject] == 'F':
            females += 1
        else:
            males += 1


    print(f"There are {len(subject_sex)} subjects, {males} males and {females} females.")
    print(f"There are: {len(sessions)} sessions, {male_sessions} male sessions and {female_sessions} female sessions.")
    print(f"The oldest subject is {max(ages)} and the youngest subject is {min(ages)}")
    print(f"The median age of subjects is {statistics.median(ages)} and the mean age is {statistics.mean(ages)}")
    print(
        f"The lowest number of sessions per patient is {min(subject_sessions.values())} and the highest is "
        f"{max(subject_sessions.values())}")
    print(f"The median number of sessions per patient is {statistics.median(subject_sessions.values())} and the mean is "
          f"{statistics.mean(subject_sessions.values())}")

    print(f"Our dataset contains {no_dementia} samples in the non-demented class, {very_mild_dementia} samples in the very-mildly-demented class, {mild_dementia} samples in the mildly-demented class, "
          f"{moderate_dementia} samples in the moderately-demented class, and {severe_dementia} samples in the severely-demented class, ")


# Takes a list of CSV rows and removes any imaging sessions
# that take place at the same age as another session.
# Returns a list of still-valid rows and the number of duplicates identified.
def strip_duplicate_imaging(rows):
    subjects = {}
    duplicate_count = 0
    for row in rows:
        subject = row[1]
        if subject in subjects:
            age = row[4]
            if age in subjects[subject]:
                duplicate_count += 1
            else:
                subjects[subject].append(age)
        else:
            age = row[4]
            subjects[subject] = [age]

    return subjects, duplicate_count


## Inputs
# folder_path: relative path to the folder of DICOM images
# center_numbers: array of MRI slice indices
## Outputs
# None: saves the resulting images to the Images directory
def convert_to_jpg(folder_path, center_numbers):
    # use the folder containing the images you want to load
    # pattern "*" means all the images in the folder
    file_names = pydicom.data.data_manager.get_files(folder_path, "*")
    # convert center_numbers to the format you will find them in the file names
    for n, number in enumerate(center_numbers):
        center_numbers[n] = "_{}_".format(number)

    # get file names that are only the frames specified in center_numbers
    center_file_names = []
    for file_name in file_names:
        if any(number in file_name for number in center_numbers):
            center_file_names.append(file_name)


    ## WIP functionality to create folders within \Images to organize created images
    # parent_dir = os.path.join(os.getcwd(),"Images")
    # new_dir = folder_path.replace('Example\\','')
    # path = os.path.join(parent_dir,new_dir)
    # print(path)
    # os.mkdir(path) # Make new folder in Images

    # save the files
    for file_name in center_file_names:
        pixel_array = dcmread(file_name).pixel_array
        new_file_name = file_name.replace('.dcm','.jpg').replace(folder_path,'').replace('\\','')
        save_path = os.path.join("Images",new_file_name)
        plt.imsave(save_path, pixel_array, cmap=plt.cm.bone) # set the color map to bone

if __name__ == '__main__':
    #filename = 'Data/AxialWithCDR18981.csv'
    #process_csv(filename)

    import numpy as np
    import matplotlib.pyplot as plt
    from pydicom import dcmread
    from pydicom.data.data_manager import get_files
    import os

    # Full path of the DICOM file is passed in base
    folder_path = 'Example\\941_S_5193\\AXIAL_T2_STAR\\2015-04-02_11_01_34.0\\I673865'
    pass_dicom = "ADNI_941_S_5193_MR_AXIAL_T2_STAR__br_raw_20160407105052903_24_S412718_I673865.dcm"

    ## Getting a specific file:
    # enter DICOM image name for pattern
    # result is a list of 1 element
    # filename = pydicom.data.data_manager.get_files(folder_path, "*")

    center_numbers = [i for i in range(18,23)] # Specifies the frames to get from the image
    # pixel_arrays = get_center_pixel_arrays(base, center_numbers)
    # for image in pixel_arrays:
    #     plt.imshow(image, cmap=plt.cm.bone)  # set the color map to bone
    #     plt.show()
    #     plt.imsave("exampledicom.jpg", image, cmap=plt.cm.bone)
    convert_to_jpg(folder_path,center_numbers)