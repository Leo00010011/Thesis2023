from utils.bag_utils import BagReview
from utils.IQA import LAP_MOD
bag_path = 'tomas\\20230511_102553.bag'

bag = BagReview(bag_path)

bag.review_frames_IQA(LAP_MOD)