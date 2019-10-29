from datasets import ExploratoryDataset
from utils import *
from models import DescriptionNet,TitleNet,get_pretrained_resnet
import torch
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon as jensenshannondistance
import sys
import pandas as pd

COLUMN_NAMES = ['id','image_class_prediction','image_probability','image_gt_distance',
                'description_class_prediction','description_probability','description_gt_distance',
                'title_class_prediction','title_probability','title_gt_distance',
                'description_title_distance','description_image_distance','title_image_distance']
## initializing data dictionary for postprocessing results
data_dict = {}
for name in COLUMN_NAMES:
    data_dict[name] = []

dataset = ExploratoryDataset(transform=get_augmentation_dict())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False, num_workers=0)
print('dataset laoding finished')

labels = dataset.y_codec.classes_
classes = dataset.y_len
desc_size = dataset.desc_len
title_size = dataset.title_len

## loading pretrained models
image_net = get_pretrained_resnet(classes)
desc_net = DescriptionNet(input_size=desc_size, num_classes=classes)
title_net = TitleNet(input_size=title_size,num_classes=classes)

image_net.load_state_dict(torch.load('../models/image_resnet_50_002.pth',map_location='cpu'))
image_net.eval()
desc_net.load_state_dict(torch.load('../models/desc_100_015.pth',map_location='cpu'))
desc_net.eval()
title_net.load_state_dict(torch.load('../models/title_100_015.pth',map_location='cpu'))
title_net.eval()
print('models loading finished')

for i, data in enumerate(dataloader, 0):
    image, desc, title, gt, item_id = data
    image_out = F.softmax(image_net(image), dim=1)
    desc_out = F.softmax(desc_net(desc), dim=1)
    title_out = F.softmax(title_net(title), dim=1)

    image_out_numpy = image_out.detach().numpy()[0]
    desc_out_numpy = desc_out.detach().numpy()[0]
    title_out_numpy = title_out.detach().numpy()[0]

    image_max = torch.max(image_out,1)
    desc_max = torch.max(desc_out,1)
    title_max = torch.max(title_out,1)

    ground_truth_vector = np.zeros(len(labels))
    ground_truth_vector[gt.numpy()[0]] = 1

    ## appending postprocessing results into dictionary
    data_dict['id'].append(item_id[0])
    data_dict['image_probability'].append(image_max[0].detach().numpy()[0])
    data_dict['title_probability'].append(title_max[0].detach().numpy()[0])
    data_dict['description_probability'].append(desc_max[0].detach().numpy()[0])

    data_dict['image_class_prediction'].append(labels[image_max[1].numpy()[0]])
    data_dict['description_class_prediction'].append(labels[desc_max[1].numpy()[0]])
    data_dict['title_class_prediction'].append(labels[title_max[1].numpy()[0]])

    data_dict['image_gt_distance'].append(jensenshannondistance(ground_truth_vector,image_out_numpy))
    data_dict['description_gt_distance'].append(jensenshannondistance(ground_truth_vector,desc_out_numpy))
    data_dict['title_gt_distance'].append(jensenshannondistance(ground_truth_vector,title_out_numpy))
    data_dict['description_title_distance'].append(jensenshannondistance(desc_out_numpy,title_out_numpy))
    data_dict['description_image_distance'].append(jensenshannondistance(desc_out_numpy,image_out_numpy))
    data_dict['title_image_distance'].append(jensenshannondistance(title_out_numpy,image_out_numpy))

    sys.stdout.write("\rPostprocessing:[Batch %d/%d]" % (i, len(dataloader)))
print(data_dict)

df = pd.DataFrame.from_dict(data_dict)
df.to_csv('../data/postprocessing.csv', index=False)

print('postprocessing results stored')