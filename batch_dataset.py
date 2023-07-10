import numpy as np
import os.path as path
import random
import pickle
import glob
from PIL import Image
import io


def get_class_count():
    class_count = np.asarray([339, 49, 9, 191, 3, 431, 4, 195, 19, 15])

    return class_count


def randomise_by_index(inputlist, idx_list):  # A helper function for later
    """
    Function to randomize an array of data
    """

    if len(inputlist) != len(idx_list):
        print("These aren't the same length")

    outputlist = []
    for i in idx_list:
        outputlist.append(inputlist[i])

    return outputlist


def multiclass_batch(class_count, batches, batch_size):
    """
    Figures out how many images of each class should be in each batch to make contents
    as consistent as possible
    """

    if path.exists("./mirabest_batch_list.p") is False:
        class_count = np.array(class_count)
        len_class = len(class_count)

        # First check: sum up the total number of files and make sure it equals no. x length of batches
        if sum(class_count) == (batches * batch_size):
            extra_count = 0

        else:
            # Any over/underflow in length should go into the test batch;
            # this will keep track of how many should go in
            extra_count = sum(class_count) - (batches * batch_size)

        # ---------------------

        test_count = np.floor(
            class_count / batches
        )  # Divide the class vector by the no. of batches and floor it
        test_count[np.where(test_count == 0)] = 1  # If any entry equals zero, set it to one instead

        if (
            sum(test_count) == batch_size and extra_count == 0
        ):  # Case where the number of test entries divides by batch size
            print("Test batch complete")

        elif sum(test_count) == batch_size and extra_count != 0:  # Case when extras are needed; WIP
            print("This bit's not set up yet")

        # Case when there's not enough entries
        elif sum(test_count) < batch_size:
            # Find how many entries are left to fill,
            # and which classes have the largest remainders when divided
            to_fill = (batch_size - sum(test_count) + extra_count).astype(int)
            remainder = class_count % batches

            # If there are fewer entries to be filled than there are classes
            if to_fill <= len_class:
                # Find the indices of the largest n entries' remainders,
                # where n is the number of entries to fill
                indices = (-remainder).argsort()[:to_fill]

                for i in range(to_fill):
                    test_count[indices[i]] = test_count[indices[i]] + 1

            # Otherwise, there are more entries to be filled than there are classes
            # If number to fill divides neatly by the number of classes,
            # test_count just gets that number added to every entry
            elif to_fill % len_class == 0:
                test_count = test_count + (to_fill / len_class)

            # If not, add an even number to each, then add the remainder to the largest remaining classes
            else:
                for i in range(int(to_fill)):
                    test_count = test_count + np.floor(to_fill / class_count)

                    last_fill = to_fill % len_class

                    indices = (-remainder).argsort()[:last_fill]

                    # Add one to the counts for each of the n largest remainders
                    for i in range(last_fill):
                        test_count[indices[i]] = test_count[indices[i]] + 1

            print("Test batch complete")

        else:  # Only remaining possibility is that there's too many entries
            # Find out how many extra entries there are,
            # and which classes have the largest remainders when divided
            to_remove = test_count - batch_size - extra_count
            remainder = class_count % batches

            # Find the indices of the smallest n entries' remainders,
            # where n is the number of entries to fill
            indices = (remainder).argsort()[:to_fill]

            no_replaced = 0

            # Check if any of the chosen classes have only one image
            while any(test_count[indices] == 1) is True:
                # If they do, find how many
                to_replace = np.where(test_count[indices]) == 1
                no_to_replace = len(to_replace)

                # Find the next largest remainders
                index_replacements = (remainder).argsort()[
                    to_fill + no_replaced : to_fill + no_replaced + no_to_replace
                ]

                # Replace the previous entries
                for i in range(no_to_replace):
                    indices[to_replace[i]] = index_replacements[i]

                # Note how many have now been replaced in case another pass is needed
                no_replaced = no_replaced + no_to_replaced

            # Now all the checks have been done, should be free to remove all necessary
            for i in range(to_fill):
                test_count[indices[i]] = test_count[indices[i]] - 1

            print("Test batch complete")

        training_count = class_count - test_count

        # Total no. of images of each class present in every batch:
        base_count = np.floor(training_count / (batches - 1))

        # Number of remaining images of each class
        extra_count = training_count % (batches - 1)

        # Check if the base count already contains batch_size images
        if sum(base_count) == batch_size:
            print("Training batches complete")

            training_count = np.array(
                [
                    base_count,
                ]
                * (batches - 1)
            )

        else:
            to_add = batch_size - sum(base_count)  # Find the number to add to each batch

            training_count = np.array(
                [
                    base_count,
                ]
                * (batches - 1)
            )

            for i in range(batches - 1):
                for j in range(to_add.astype(int)):
                    # Check which (if any) of the classes have images to use
                    index_options = np.asarray(np.nonzero(extra_count)[0])

                    # Select a random one of these indices
                    index_choice = random.choice(index_options)

                    # Add one to training_count in relevant entry; subtract one from extra_count in relevant entry
                    training_count[i, index_choice] = training_count[i, index_choice] + 1
                    extra_count[index_choice] = extra_count[index_choice] - 1

            print("Training batches complete")

        batch_list = np.vstack((training_count, test_count))

        pickle.dump(batch_list, open("./mirabest_batch_list.p", "wb"))

    else:
        batch_list = pickle.load(open("./mirabest_batch_list.p", "rb"))

    return batch_list


def generate_index_list(class_count):
    """Generates or loads a fixed list of indices to reorder the dataset with"""

    if path.exists("./mirabest_index_list.p") is False:
        class_count = np.array(class_count)

        index_list = np.empty([np.sum(class_count)])

        startpoint = 0

        for i in range(len(class_count)):
            index_subset = np.arange(0, class_count[i], 1)

            random.shuffle(index_subset)

            index_list[startpoint : startpoint + class_count[i]] = index_subset

            startpoint = startpoint + class_count[i]

        pickle.dump(index_list, open("./mirabest_index_list.p", "wb"))

    else:
        index_list = pickle.load(open("./mirabest_index_list.p", "rb"))

    return index_list


def build_dataset(file_loc, bind_dir, nbatch, pbatch):
    """file_loc is where the images are, bind_dir where the batched dataset should go"""

    class_count = get_class_count()

    index_list = generate_index_list(nbatch * pbatch)
    batch_count = multiclass_batch(class_count, nbatch, pbatch)

    # ---------------------

    cl0_files = np.sort(np.array(glob.glob(file_loc + "100*.png")))
    cl1_files = np.sort(np.array(glob.glob(file_loc + "102*.png")))
    cl2_files = np.sort(np.array(glob.glob(file_loc + "104*.png")))
    cl3_files = np.sort(np.array(glob.glob(file_loc + "110*.png")))
    cl4_files = np.sort(np.array(glob.glob(file_loc + "112*.png")))
    cl5_files = np.sort(np.array(glob.glob(file_loc + "200*.png")))
    cl6_files = np.sort(np.array(glob.glob(file_loc + "201*.png")))
    cl7_files = np.sort(np.array(glob.glob(file_loc + "210*.png")))
    cl8_files = np.sort(np.array(glob.glob(file_loc + "300*.png")))
    cl9_files = np.sort(np.array(glob.glob(file_loc + "310*.png")))

    n_cl0 = len(cl0_files)
    n_cl1 = len(cl1_files)
    n_cl2 = len(cl2_files)
    n_cl3 = len(cl3_files)
    n_cl4 = len(cl4_files)
    n_cl5 = len(cl5_files)
    n_cl6 = len(cl6_files)
    n_cl7 = len(cl7_files)
    n_cl8 = len(cl8_files)
    n_cl9 = len(cl9_files)

    # assert ((n_cl0+n_cl1+n_cl2+n_cl3+n_cl4+n_cl5+n_cl6+n_cl7+n_cl8+n_cl9)>=(pbatch*nbatch)),
    # 'Not enough samples available to fill '+str(nbatch)+' batches of '+str(pbatch)

    stop_index = np.zeros(10, dtype=int)
    stop_index[0] = n_cl0
    stop_index[1] = stop_index[0] + n_cl1
    stop_index[2] = stop_index[1] + n_cl2
    stop_index[3] = stop_index[2] + n_cl3
    stop_index[4] = stop_index[3] + n_cl4
    stop_index[5] = stop_index[4] + n_cl5
    stop_index[6] = stop_index[5] + n_cl6
    stop_index[7] = stop_index[6] + n_cl7
    stop_index[8] = stop_index[7] + n_cl8
    stop_index[9] = stop_index[8] + n_cl9

    cl0_shuffle = index_list[0 : stop_index[0]]
    cl1_shuffle = index_list[stop_index[0] : stop_index[1]]
    cl2_shuffle = index_list[stop_index[1] : stop_index[2]]
    cl3_shuffle = index_list[stop_index[2] : stop_index[3]]
    cl4_shuffle = index_list[stop_index[3] : stop_index[4]]
    cl5_shuffle = index_list[stop_index[4] : stop_index[5]]
    cl6_shuffle = index_list[stop_index[5] : stop_index[6]]
    cl7_shuffle = index_list[stop_index[6] : stop_index[7]]
    cl8_shuffle = index_list[stop_index[7] : stop_index[8]]
    cl9_shuffle = index_list[stop_index[8] : stop_index[9]]

    # ---------------------

    count = np.zeros(10)

    for batch in range(nbatch):
        if batch == (nbatch - 1):
            # the last batch is the test batch:
            oname = "test_batch"
            batch_label = "testing batch 1 of 1"
            pbatch = 156
        else:
            # everything else is a training batch:
            oname = "data_batch_" + str(batch + 1)
            batch_label = "training batch " + str(batch + 1) + " of " + str(nbatch - 1)

        # create empty arrays for the batches:
        labels = []
        filedata = []
        data = []
        filenames = []

        # Class 0
        for i in range(int(count[0]), int(count[0] + batch_count[batch, 0])):
            filename = cl0_files[int(cl0_shuffle[i])]
            filenames.append(filename)

            labels.append(0)

            im = Image.open(filename)
            im = np.array(im)
            filedata = np.array(list(im), np.uint8)
            data.append(filedata)

        count[0] = count[0] + batch_count[batch, 0]

        # Class 1
        for i in range(int(count[1]), int(count[1] + batch_count[batch, 1])):
            filename = cl1_files[int(cl1_shuffle[i])]
            filenames.append(filename)

            labels.append(1)

            im = Image.open(filename)
            im = np.array(im)
            filedata = np.array(list(im), np.uint8)
            data.append(filedata)

        count[1] = count[1] + batch_count[batch, 1]

        # Class 2
        for i in range(int(count[2]), int(count[2] + batch_count[batch, 2])):
            filename = cl2_files[int(cl2_shuffle[i])]
            filenames.append(filename)

            labels.append(2)

            im = Image.open(filename)
            im = np.array(im)
            filedata = np.array(list(im), np.uint8)
            data.append(filedata)

        count[2] = count[2] + batch_count[batch, 2]

        # Class 3
        for i in range(int(count[3]), int(count[3] + batch_count[batch, 3])):
            filename = cl3_files[int(cl3_shuffle[i])]
            filenames.append(filename)

            labels.append(3)

            im = Image.open(filename)
            im = np.array(im)
            filedata = np.array(list(im), np.uint8)
            data.append(filedata)

        count[3] = count[3] + batch_count[batch, 3]

        # Class 4
        if batch_count[batch, 4] != 0:
            for i in range(int(count[4]), int(count[4] + batch_count[batch, 4])):
                filename = cl4_files[int(cl4_shuffle[i])]
                filenames.append(filename)

                labels.append(4)

                im = Image.open(filename)
                im = np.array(im)
                filedata = np.array(list(im), np.uint8)
                data.append(filedata)

            count[4] = count[4] + batch_count[batch, 4]

        # Class 5
        for i in range(int(count[5]), int(count[5] + batch_count[batch, 5])):
            filename = cl5_files[int(cl5_shuffle[i])]
            filenames.append(filename)

            labels.append(5)

            im = Image.open(filename)
            im = np.array(im)
            filedata = np.array(list(im), np.uint8)
            data.append(filedata)

        count[5] = count[5] + batch_count[batch, 5]

        # Class 6
        if batch_count[batch, 6] != 0:
            for i in range(int(count[6]), int(count[6] + batch_count[batch, 6])):
                filename = cl6_files[int(cl6_shuffle[i])]
                filenames.append(filename)

                labels.append(6)

                im = Image.open(filename)
                im = np.array(im)
                filedata = np.array(list(im), np.uint8)
                data.append(filedata)

            count[6] = count[6] + batch_count[batch, 6]

        # Class 7
        for i in range(int(count[7]), int(count[7] + batch_count[batch, 7])):
            filename = cl7_files[int(cl7_shuffle[i])]
            filenames.append(filename)

            labels.append(7)

            im = Image.open(filename)
            im = np.array(im)
            filedata = np.array(list(im), np.uint8)
            data.append(filedata)

        count[7] = count[7] + batch_count[batch, 7]

        # Class 8
        for i in range(int(count[8]), int(count[8] + batch_count[batch, 8])):
            filename = cl8_files[int(cl8_shuffle[i])]
            filenames.append(filename)

            labels.append(8)

            im = Image.open(filename)
            im = np.array(im)
            filedata = np.array(list(im), np.uint8)
            data.append(filedata)

        count[8] = count[8] + batch_count[batch, 8]

        # Class 9
        for i in range(int(count[9]), int(count[9] + batch_count[batch, 9])):
            filename = cl9_files[int(cl9_shuffle[i])]
            filenames.append(filename)

            labels.append(9)

            im = Image.open(filename)
            im = np.array(im)
            filedata = np.array(list(im), np.uint8)
            data.append(filedata)

        count[9] = count[9] + batch_count[batch, 9]

        # randomise data in batch:
        idx_list = range(0, pbatch)
        labels = randomise_by_index(labels, idx_list)
        data = randomise_by_index(data, idx_list)
        filenames = randomise_by_index(filenames, idx_list)

        # create dictionary of batch:
        dict = {"batch_label": batch_label, "labels": labels, "data": data, "filenames": filenames}

        # write pickled output:
        with io.open(bind_dir + oname, "wb") as f:
            pickle.dump(dict, f)

        # end batch loop

    # total number of samples per batch: [modified 20190925]
    pbatch = 157

    # label names: [modified 20190415]
    label_names = ["100", "102", "104", "110", "112", "200", "201", "210", "300", "310"]

    # length of data arrays [npix x npix x rgb = 150 x 150 x 1 = 22500] [modified 20190415]
    nvis = 22500

    # now write the meta data file:
    oname = "batches.meta"

    # create dictionary of batch:
    dict = {
        "num_cases_per_batch": pbatch,
        "label_names": label_names,
        "num_vis": nvis,
    }

    # ------------------------------------
    # write pickled output:
    with io.open(bind_dir + oname, "wb") as f:
        pickle.dump(dict, f)
