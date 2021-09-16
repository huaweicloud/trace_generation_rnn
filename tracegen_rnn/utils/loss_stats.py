"""Helper class to track the loss statistics as we go through
training.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""


class LossStats():
    """A class to hold, and reset as needed, the loss stats, during
    training or testing.

    """
    def __init__(self):
        """Initialize all our running totals to zero."""
        self.tot_loss = 0
        self.tot_examples = 0

    def update(self, num, loss):
        """Given we've processed num examples, and observed an average loss of
        loss, update our totals.

        """
        if num == 0:
            return
        self.tot_loss += loss * num
        self.tot_examples += num

    def get_tot_examples(self):
        """Return total number of examples processed since beginning."""
        return self.tot_examples

    def overall_loss(self):
        """Calculate and return the overall loss."""
        return self.tot_loss / self.tot_examples
