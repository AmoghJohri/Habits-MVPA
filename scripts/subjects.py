"""scripts.subjects
====================

Utilities for managing and validating the study participant lists used
throughout this repository. This module centralizes canonical subject ID
sets (fMRI cohort, healthy controls, patients, training/held-out splits)
and provides helpers to convert and validate those lists.

Key functions
- `get_fmri_subjects()` -> list[Subject]: all participants with complete fMRI
    data used in analyses.
- `get_hc()` / `get_patients()` -> list[Subject]: healthy controls and patient
    subsets.
- `get_subjects()` / `get_test_subjects()` / `get_test_subjects_2()` -> lists
    used for training and held-out evaluations.
- `get_all_analysis_participants()` -> list[Subject]: union of training and
    test sets.
- `get_subids(subjects)` -> list[int]: extract numeric IDs from Subject
    objects.
- `validate()` -> runs internal consistency checks and prints counts.

Usage
-----
Import and use the helpers in analysis scripts or call the module directly
to run the validation checks::

        from scripts.subjects import Subject
        sub_list = Subject.get_fmri_subjects()
        ids = Subject.get_subids(sub_list)

Or run from the command line to execute `validate()` which asserts the
expected subset relationships and prints subject counts::

        python -m scripts.subjects
"""

class Subject:
    def __init__(self, subid):
        self.subid = subid    # subject-ID 

    @staticmethod
    def get_fmri_subjects() -> list:
        # List of all subjects that completed both fMRI runs for the habit task + structural scans
        subids = [2, 3, 4, 6, 7, 8, 
                  10, 12, 13, 14, 16, 17, 18, 19, 
                  20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                  30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
                  40, 41, 42, 44, 45, 46, 47, 48, 49, 
                  50, 51, 52, 53, 54, 56, 57, 58, 59, 
                  60, 61, 63, 64, 65, 66, 67, 68, 69, 
                  70, 71, 72, 74, 76, 77, 78, 
                  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 
                  90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 
                  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
                  110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 
                  120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 
                  130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 
                  140, 141, 142, 143, 145, 146, 147, 148, 149, 
                  150, 151, 153, 154, 155, 156, 157, 158, 159, 
                  160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 
                  171, 172, 173, 174, 175, 176, 177, 179, 
                  180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 
                  190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 
                  200, 201, 202, 203, 204, 205, 206, 207, 209, 
                  210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 
                  220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 
                  230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 
                  241, 242, 244, 245, 246, 247, 248, 249, 
                  250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 
                  260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 
                  270, 271]
        subids = sorted(subids)
        return [Subject(subid) for subid in subids]
    
    @staticmethod
    def get_hc() -> list:
        # All healthy controls within the fMRI subjects subset
        subids = [2, 3, 4, 6, 7, 8, 
                  10, 12, 13, 14, 16, 17, 18, 19, 
                  20, 21, 22, 23, 25, 28, 29, 
                  30, 31, 32, 33, 34, 35, 36, 37, 38, 
                  41, 42, 46, 48, 49, 
                  50, 51, 52, 53, 54, 57, 58, 59, 
                  60, 61, 63, 64, 65, 66, 69, 
                  70, 71, 72, 74, 76, 77, 78, 
                  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 
                  90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 
                  101, 109, 112, 114, 115, 116, 117, 118, 119, 
                  120, 122, 123, 125, 126, 127, 128, 129, 
                  130, 131, 134, 136, 138, 
                  140, 141, 142, 143, 145, 146, 148, 149, 
                  158, 159, 
                  161, 163, 164, 166, 167, 168, 169, 
                  171, 172, 173, 174, 176, 177, 179, 
                  180, 181, 182, 183, 184, 185, 186, 188, 189, 
                  190, 191, 192, 195, 196, 197, 198, 
                  202, 203, 204, 205, 206, 207, 209, 
                  210, 212, 213, 214, 215, 216, 217, 218, 219, 
                  221, 222, 224, 227, 228, 229, 
                  230, 232, 233, 234, 237, 238, 
                  242, 244, 245, 246, 247, 248, 
                  250, 251, 252, 253, 255, 256, 
                  261, 262, 265, 268, 269, 
                  270]
        subids = sorted(subids)
        return [Subject(subid) for subid in subids]
    
    @staticmethod
    def get_patients() -> list:
        # All patients within the fMRI subjects subset
        subids = [24, 26, 27, 
                  39, 
                  40, 44, 45, 47, 
                  56, 
                  67, 68, 
                  100, 102, 103, 104, 105, 106, 107, 108, 
                  110, 111, 113, 
                  121, 124, 
                  132, 133, 135, 137, 139, 
                  147, 
                  150, 151, 153, 154, 155, 156, 157, 
                  160, 162, 165, 
                  175, 
                  187, 
                  193, 194, 199, 
                  200, 201, 
                  211, 
                  220, 223, 225, 226, 
                  231, 235, 236, 239, 
                  241, 249, 
                  254, 257, 258, 259, 
                  260, 263, 264, 266, 267, 
                  271]
        subids = sorted(subids)
        return [Subject(subid) for subid in subids]
    
    @staticmethod
    def get_patient_subjects() -> list:
        # All patient subjects within the fMRI subjects subset
        subids = [24, 27, 
                  39, 
                  40, 44, 45, 47, 
                  56, 
                  67, 68, 
                  100, 103, 106, 107, 
                  110, 111, 113, 
                  124, 
                  133, 135, 137, 139, 
                  147, 
                  150, 151, 153, 155, 156, 157, 
                  160, 162, 165, 
                  175, 
                  187, 
                  194, 199, 
                  200, 201, 211, 
                  220, 223, 225, 226, 
                  235, 236, 239, 
                  241, 
                  254, 257, 258, 259, 
                  260, 263, 267, 
                  271]
        subids = sorted(subids)
        return [Subject(subid) for subid in subids]
    
    @staticmethod
    def get_subjects() -> list:
        # N = 108 Healthy Control Subjects, used for initial anaylis and training the classifier
        # Initial Cohort
        subids = sorted([2, 3, 4, 7, 
                         10, 12, 13, 19, 
                         20, 22, 23, 25, 
                         30, 31, 32, 34, 36, 37, 38, 
                         41, 46, 48, 49, 
                         50, 51, 52, 53, 54, 57, 58, 59, 
                         60, 61, 63, 64, 65, 66, 69, 
                         70, 72, 74, 76, 77, 
                         80, 81, 82, 83, 84, 85, 87, 88, 89, 
                         91, 92, 93, 94, 95, 96, 97, 98, 99, 
                         101, 109, 
                         112, 114, 115, 116, 117, 118, 119, 
                         120, 125, 126, 128, 129, 
                         130, 131, 134, 136, 138, 
                         140, 142, 143, 145, 149, 
                         158, 159, 
                         161, 163, 164, 166, 168, 169, 
                         171, 173, 174, 176, 177, 179, 
                         180, 181, 184, 185, 186, 188, 
                         197, 
                         202, 
                         218])
        subids = sorted(subids)
        return [Subject(subid) for subid in subids]
    
    @staticmethod
    def get_test_subjects() -> list:
        # N = 36 Healthy Control Test Subjects, used for final evaluation of classifier
        # Held-Out Cohort
        subids = [190, 191, 192, 195, 
                  205, 206, 207, 209, 
                  210, 214, 215, 216, 217, 219, 
                  221, 224, 228, 229, 
                  230, 232, 233, 234, 238, 
                  245, 246, 247, 248, 
                  251, 252, 253, 256, 
                  261, 262, 268, 269, 
                  270]
        subids = sorted(subids)
        return [Subject(subid) for subid in subids]
    
    @staticmethod
    def get_test_subjects_2() -> list:
        # N = 55 Patient Test Subjects, used for final evaluation of classifier in a heterogeneous clinical popualtion
        # Held-Out Patient Cohort
        subids = [24, 27, 
                  39, 
                  40, 44, 45, 47, 
                  56, 
                  67, 68, 
                  100, 103, 106, 107, 
                  110, 111, 113, 
                  124, 
                  133, 135, 137, 139, 
                  147, 
                  150, 151, 153, 155, 156, 157, 
                  160, 162, 165, 
                  175, 
                  187, 
                  194, 199, 
                  200, 201, 
                  211, 
                  220, 223, 225, 226, 
                  235, 236, 239, 
                  241, 
                  254, 257, 258, 259, 
                  260, 263, 267, 
                  271]
        subids = sorted(subids)
        return [Subject(subid) for subid in subids]
    
    @staticmethod
    def get_all_analysis_participants() -> list[int]:
        # Training subjects + Test subjects (Healthy Controls + Patients)
        training_subs = Subject.get_subjects()
        test_subs     = Subject.get_test_subjects()
        test_subs_2   = Subject.get_test_subjects_2()
        all_subs      = sorted(training_subs + test_subs + test_subs_2, key=lambda x: x.subid)
        return all_subs
    
    @staticmethod
    # Helper function to extract subject IDs from a list of Subject objects
    def get_subids(subjects) -> list[int]:
        return [sub.subid for sub in subjects]
    
    @staticmethod
    def validate():
        """
        1. Healty controls should belong in the fMRI subjects subset
        2. Patients should belong in the fMRI subjects subset
        3. Healthy controls and patients should not overlap
        4. Training subjects should belong in the Healthy controls subset
        5. Test subjects should belong in the Healthy controls subset
        6. Test (Patient) subjects should belong in the Patients subset
        7. Training, Test, Test (Patients) subjects should not overlap
        8. Prints out any violations of the above conditions, and the number of subjects in each category
        """
        # Get all relevant subject subsets
        fmri_subs  = Subject.get_fmri_subjects()
        hc_subs    = Subject.get_hc()
        pt_subs    = Subject.get_patients()
        train_subs = Subject.get_subjects()
        test_subs  = Subject.get_test_subjects()
        test2_subs = Subject.get_test_subjects_2()
        # Extract subject IDs for easy comparison
        fmri_ids   = set(Subject.get_subids(fmri_subs))
        hc_ids     = set(Subject.get_subids(hc_subs))
        pt_ids     = set(Subject.get_subids(pt_subs))
        train_ids  = set(Subject.get_subids(train_subs))
        test_ids   = set(Subject.get_subids(test_subs))
        test2_ids  = set(Subject.get_subids(test2_subs))
        # Validate subset relationships
        assert hc_ids.issubset(fmri_ids),       "Some healthy controls are not in the fMRI subjects subset"
        assert pt_ids.issubset(fmri_ids),       "Some patients are not in the fMRI subjects subset"
        assert hc_ids.isdisjoint(pt_ids),       "Healthy controls and patients overlap"
        assert train_ids.issubset(hc_ids),      "Some training subjects are not in the healthy controls subset"
        assert test_ids.issubset(hc_ids),       "Some test subjects are not in the healthy controls subset"
        assert test2_ids.issubset(pt_ids),      "Some test (patient) subjects are not in the patients subset"
        assert train_ids.isdisjoint(test_ids),  "Training and Test subjects overlap"
        assert train_ids.isdisjoint(test2_ids), "Training and Test (Patient) subjects overlap"
        assert test_ids.isdisjoint(test2_ids),  "Test and Test (Patient) subjects overlap"
        # Print counts
        print(f"Number of fMRI subjects: {len(fmri_ids)}")
        print(f"Number of healthy controls: {len(hc_ids)}")
        print(f"Number of patients: {len(pt_ids)}")
        print(f"Number of training subjects: {len(train_ids)}")
        print(f"Number of test subjects: {len(test_ids)}")
        print(f"Number of test (patient) subjects: {len(test2_ids)}")
        
def main():
    Subject.validate()

if __name__ == "__main__":
    main()