from utils.bag_utils import BagReview
from utils.DMQA import entropy_DMQA
bag_path = 'tomas\\20230511_102553.bag'

bag = BagReview(bag_path)

bag.review_frames_DMQA(entropy_DMQA)