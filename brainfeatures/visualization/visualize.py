import os

import matplotlib.pyplot as plt
import numpy as np


ELECTRODE_LOCATIONS = {
    'PZ': [50, 30],
    'CZ': [50, 50],
    'FZ': [50, 70],

    'O1': [40, 10],
    'O2': [60, 10],

    'T5': [13, 25],
    'T6': [87, 25],

    'P3': [32, 30],
    'P4': [68, 30],

    'F3': [32, 70],
    'F4': [68, 70],

    'F7': [13, 75],
    'F8': [87, 75],

    'FP1': [40, 90],
    'FP2': [60, 90],

    'A1': [-7, 52],
    'T3': [11, 50],
    'C3': [30, 50],
    'C4': [70, 50],
    'T4': [89, 50],
    'A2': [107, 52],
}
ELECTRODE_NAMES = sorted(ELECTRODE_LOCATIONS.keys())


# TODO: plot electrode importances as colormap on the head scheme


# TODO: plot correlation maps of different domains, create ticks and ticklabels
def plot_feature_correlations(data, title='', xticks=None, xticklabels=None,
                              yticks=None, yticklabels=None, offset=1):
    xticklabels = np.array(xticklabels)
    yticklabels = np.array(yticklabels)
    plt.figure(figsize=(12, 12))
    plt.title(title, fontsize=20)

    plt.imshow(
        data,
        cmap="RdBu_r",
        interpolation=None,
        vmin=-1,
        vmax=1,
        aspect="equal"
    )

    ax = plt.gca()
    ax.tick_params(
        axis=u'both',
        which=u'major',
        length=5,
        direction=u"inout"
    )

    if xticks is not None and xticklabels is not None:
        plt.yticks(
            xticks,
            xticklabels[xticks],
            fontsize=15
        )
    else:
        plt.yticks([])
    if yticks is not None and yticklabels is not None:
        plt.xticks(
            yticks,
            yticklabels[yticks - offset],
            rotation=90,
            fontsize=15
        )
    else:
        plt.xticks([])

    cb = plt.colorbar(
        fraction=0.046,
        pad=0.04
    )
    cb.ax.tick_params(labelsize=15)
    plt.grid(False)


def plot_mean_feature_importances_spatial(mean_importances, feature_labels,
                                          out_dir=None, n_most_important=5,
                                          fontsize=10):
    """ plot the feature importances on a scheme of the head to add spatial
    relationship.
    :param importances: the feature importances as returned by the RF. shape
         n_folds x n_features
    :param out_dir: the output directory where the plots are saved to
    :param feature_labels: the list of features
    :param n_most_important: the number of features that should be analyzed
    """
    plt.figure(figsize=(15, 10))
    plt.rcParams['axes.facecolor'] = 'white'
    # fontsize=10

    elecs = []
    for feature_label in feature_labels:
        splits = feature_label.split("_")
        n_splits = len(splits)
        elec = splits[-1]
        # check if we know this electrode already, make sure its not age/gender
        if n_splits > 2 and elec not in elecs:
            elecs.append(elec)

    electrode_to_feature_labels = {}
    for elec in elecs:
        electrode_to_feature_labels.update({elec: []})

    # mean_importances = np.mean(importances, axis=0)

    # assign the mean importances to the respective electrode
    for feature_label_id, feature_label in enumerate(feature_labels):
        for electrode_name in ELECTRODE_NAMES:
            if electrode_name in feature_label.split('_')[-1]:
                electrode_to_feature_labels[electrode_name].append(
                    feature_label_id)

    new_labels = []
    for feature_label in feature_labels:
        if "plv" in feature_label:
            new_labels.append(feature_label)
        else:
            new_labels.append('_'.join(feature_label.split('_')[:-1]))
    feature_labels = new_labels

    # __________________________________________________________________________
    # offset due to aligning strings at the bottom left corner
    offset_x = 7.7
    offset_y = -6.3
    # the head
    ax = plt.gca()
    c1 = plt.Circle((50+offset_x, 50+offset_y), 50, fill=False, color='black',
                    alpha=.5)
    ax.add_artist(c1)
    #c2 = plt.Circle((50+offset_x, 50+offset_y), 40, fill=False, color='black',
    #                alpha=.5)
    #ax.add_artist(c2)
    # the nose
    plt.plot([47+offset_x, 50+offset_x, 53+offset_x],
             [100+offset_y, 103+offset_y, 100+offset_y],
             color='black', alpha=.5, linewidth=.1)
    # from ear to ear
    #plt.plot([0+offset_x, 100+offset_x], [50+offset_y, 50+offset_y],
    #         color='black', alpha=.5, linewidth=.1)
    # from nasion to inion
    #plt.plot([50+offset_x, 50+offset_x], [0+offset_y, 100+offset_y],
    #         color='black', alpha=.5, linewidth=.1)

    # if n_most_important == 3:
    #     colors = ['red', 'orangered', 'orange']
    # else:
    #     color_map = plt.get_cmap('nipy_spectral')
    #     colors = [color_map(i) for i in np.linspace(0, 1, n_most_important)]

    # put the n most important feature of a electrode at the respective position
    # if only patient features are used, this will crash since the
    # electrode_to_feature_labels map will not contain any entry. this is
    # because "age" and "sex" feature_label does not contain any electrode
    for electrode in ELECTRODE_NAMES:
        if not electrode_to_feature_labels[electrode]:
            continue
        zipped = zip(mean_importances[
                         np.asarray(electrode_to_feature_labels[electrode])],
                     electrode_to_feature_labels[electrode])
        sorted_zipped = sorted(zipped, reverse=True)
        sorted_mean_imp, sorted_mean_imp_ids = zip(*sorted_zipped)
        most_important_mean_imp = sorted_mean_imp[:n_most_important]
        most_important_mean_imp_ids = sorted_mean_imp_ids[:n_most_important]

        [x, y] = ELECTRODE_LOCATIONS[electrode]
        for i in range(min(n_most_important,
                           len(electrode_to_feature_labels['O1']))):
            # print(most_important_mean_imp[i])
            # print(feature_labels[most_important_mean_imp_ids[i]])
            name = feature_labels[most_important_mean_imp_ids[i]]
            value = str(most_important_mean_imp[i])[:7]
            # plot the electrode in the background
            # plt.text(x, y - 10, electrode, fontsize=40, color='blue',
            #          alpha=.03)
            # name = '_'.join(name.split('_')[1:])
            if "lyauponov" in name:
                name = "time_lyauponov_exp"

            if "sync_" in name:
                name = name.replace("sync_", "")
            if "boundedvar" in name:
                name = name.replace("fft_", "")
            if n_most_important == 3:
                # plot the n most important features together with their
                # importances at their location
                plt.text(x, y-i*5.5, name, fontsize=fontsize, fontweight='bold',
                         color='black', alpha=1-.1*i)
                plt.text(x, y-(i+.53)*5.5, str(value),  fontsize=fontsize,
                         color='black', alpha=1-.3*i)
            else:
                # plt.text(x, y-1-i*2.5, name + ' ' + value,
                #          fontsize=fontsize-1, fontweight='bold',
                #          color='black', alpha=1-.15*i)
                plt.text(x, y - 1 - i * 2.5, name, fontsize=fontsize - 1,
                         fontweight='bold', color='black', alpha=1 - .15 * i)

    title = [str(n_most_important), 'most', 'important', 'features', 'per',
             'electrode']
    # plt.title(' '.join(title))
    plt.xlim([-10, 125])
    plt.xticks([])
    plt.ylim([-10, 105])
    plt.yticks([])
    plt.tight_layout()

    plt.show()
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, '_'.join(title) + '.png'), dpi=400)
        plt.savefig(os.path.join(out_dir, '_'.join(title) + '.pdf'))
    plt.close('all')


# TODO: re-check this
# TODO: this should not do any more computations
def plot_scaled_mean_importances(list_of_properties_and_name,
                                 mean_feature_importances, feature_labels,
                                 out_dir=None):
    plt.figure(figsize=(7*len(list_of_properties_and_name), 5))
    for j, (properties, property_name) in \
            enumerate(list_of_properties_and_name):
        mean_property_importances = []
        for property_ in properties:
            importances = [feature_importance for i, feature_importance in
                           enumerate(mean_feature_importances)
                           if property_ in feature_labels[i]]
            mean_property_importances.append(np.mean(importances))

        scaled_mean_importance = mean_property_importances/np.sum(
            mean_property_importances)
        plt.subplot(1, len(list_of_properties_and_name), j+1)
        plt.bar(range(len(properties)), scaled_mean_importance)
        plt.xticks(range(len(properties)), properties, rotation=90)
        plt.ylabel("importance")
        plt.xlabel(property_name)

    if out_dir is not None:
        plt.savefig(out_dir, "importances.png", dpi=400)


def histogram(df_of_ages_genders_and_pathology_status, train_or_eval, alpha=.5,
              fs=24, ylim=20, bins=np.linspace(0, 100, 101), out_dir=None,
              show_title=True):
    df = df_of_ages_genders_and_pathology_status
    male_df = df[df["genders"] == 0]
    female_df = df[df["genders"] == 1]

    male_abnormal_df = male_df[male_df["pathologicals"] == 1]
    male_normal_df = male_df[male_df["pathologicals"] == 0]
    female_abnormal_df = female_df[female_df["pathologicals"] == 1]
    female_normal_df = female_df[female_df["pathologicals"] == 0]

    f, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, sharex=False,
                                 figsize=(15, 18))
    if show_title:
        plt.suptitle(train_or_eval+" histogram", y=.9, fontsize=fs+5)

    ax1.hist(x=male_normal_df["ages"], bins=bins, alpha=alpha, color="green",
             orientation="horizontal",
             label="normal ({:.1f}%)".format(len(male_normal_df) /
                                             len(male_df) * 100))

    ax1.hist(x=male_abnormal_df["ages"], bins=bins, alpha=alpha, color="blue",
             orientation="horizontal",
             label="pathological ({:.1f}%)".format(len(male_abnormal_df) /
                                                   len(male_df) * 100))

    ax1.axhline(np.mean(male_df["ages"]), color="black",
                # label="mean age {:.2f} $\pm$ {:.2f}".format(
                #     np.mean(male_df["age"]), np.std(male_df["age"])))
                label="mean age {:.1f} ($\pm$ {:.1f})"
                .format(np.mean(male_df["ages"]), np.std(male_df["ages"])))
    ax1.barh(np.mean(male_df["ages"]), height=2 * np.std(male_df["ages"]),
             width=ylim, color="black", alpha=.25)
    ax1.set_xlim(0, ylim)

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [2, 1, 0]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    ax1.legend(fontsize=fs, loc="lower left")
    ax1.set_title("male ({:.1f}%)".format(100 * float(len(male_df) / len(df))),
                  fontsize=fs, loc="left", y=.95, x=.05)
    ax1.invert_xaxis()

    # second axis
    ax2.hist(x=female_normal_df["ages"], bins=bins, alpha=alpha, color="orange",
             orientation="horizontal",
             label="normal ({:.1f}%)".format(len(female_normal_df) /
                                             len(female_df) * 100))

    ax2.hist(x=female_abnormal_df["ages"], bins=bins, alpha=alpha, color="red",
             orientation="horizontal",
             label="pathological ({:.1f}%)".format(len(female_abnormal_df) /
                                                   len(female_df) * 100))

    ax2.axhline(np.mean(female_df["ages"]), color="black", linestyle="--",
                # label="mean age {:.2f} $\pm$ {:.2f}"
                # .format(np.mean(female_df["age"]), np.std(female_df["age"])))
                label="mean age {:.1f} ($\pm$ {:.1f})"
                .format(np.mean(female_df["ages"]), np.std(female_df["ages"])))
    ax2.barh(np.mean(female_df["ages"]), height=2 * np.std(female_df["ages"]),
             width=ylim, color="black",
             alpha=.25)
    ax2.legend(fontsize=fs, loc="lower right")
    ax2.set_xlim(0, ylim)
    # ax1.invert_yaxis()
    ax2.set_title("female ({:.1f}%)".format(100 * len(female_df) / len(df)),
                  fontsize=fs, loc="right", y=.95, x=.95)  # , y=.005)

    plt.ylim(0, 100)
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.set_ylabel("age [years]", fontsize=fs)
    ax1.set_xlabel("count", fontsize=fs, x=1)
    # ax1.yaxis.set_label_coords(-.025, 0)
    plt.yticks(np.linspace(0, 100, 11), fontsize=fs - 5)
    ax1.tick_params(labelsize=fs - 5)
    ax2.tick_params(labelsize=fs - 5)
    # plt.savefig("tuh-abnormal-eeg-corpus-train-age-pyramid.pdf",
    #             bbox_inches="tight")
    if out_dir is not None:
        plt.savefig(out_dir+"tuh_{}.pdf".format(train_or_eval),
                    bbox_inches="tight")
