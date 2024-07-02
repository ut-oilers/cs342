import torch


class NearestNeighborClassifier:
    """
    A class to perform nearest neighbor classification.
    """

    def __init__(self, x: list[list[float]], y: list[float]):
        """
        Store the data and labels to be used for nearest neighbor classification.
        You do not have to modify this function, but you will need to implement the functions it calls.

        Args:
            x: list of lists of floats, data
            y: list of floats, labels
        """
        self.data, self.label = self.make_data(x, y)
        self.data_mean, self.data_std = self.compute_data_statistics(self.data)
        self.data_normalized = self.input_normalization(self.data)

    @classmethod
    def make_data(cls, x: list[list[float]], y: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Warmup: Convert the data into PyTorch tensors.
        Assumptions:
        - len(x) == len(y)

        Args:
            x: list of lists of floats, data
            y: list of floats, labels

        Returns:
            tuple of x and y both torch.Tensor's.
        """
        return tuple[torch.as_tensor(x), torch.as_tensor(y)]


    ##def __call__(self, x: torch.Tensor) -> torch.Tensor: #auto added
        raise NotImplementedError

    @classmethod
    def compute_data_statistics(cls, x: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Compute the mean and standard deviation of the data.
        Each row denotes a single data point.

        Args:
            x: 2D tensor data shape = [N, D]

        Returns:
            tuple of mean and standard deviation of the data.
            Both should have a shape [1, D]
        """
        return tuple[torch.tensor(x.mean(dim=0)), torch.tensor(x.std(dim=0))]

    #def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def input_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input x using the mean and std computed from the data in __init__

        Args:
            x: 1D or 2D tensor shape = [D] or [N, D]

        Returns:
            normalized 2D tensor shape = x.shape
        """
        return torch.as_tensor((x - self.data_mean) / self.data_std)

    def get_nearest_neighbor(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Find the input x's nearest neighbor and the corresponding label.

        Args:
            x: 1D tensor shape = [D]

        Returns:
            tuple of the nearest neighbor data point [D] and its label [1]
        """
        #raise NotImplementedError
        x = self.input_normalization(x)
        idx = torch.argmin(torch.norm(x - self.data_normalized, dim=1))  # Implement me:
        return tuple[torch.as_tensor(self.data[idx]), torch.as_tensor(self.label[idx])]

    def get_k_nearest_neighbor(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor]:
        """
        Find the k-nearest neighbors of input x from the data.

        Args:
            x: 1D tensor shape = [D]
            k: int, number of neighbors

        Returns:
            tuple of the k-nearest neighbors data points and their labels
            data points will be size (k, D)
            labels will be size (k,)
        """
        #raise NotImplementedError
        x = self.input_normalization(x)

        idx = torch.argsort(torch.norm(x - self.data_normalized, dim=1))[:k]  # Implement me:
        return tuple[torch.as_tensor(self.data[idx]), torch.as_tensor(self.label[idx])]

    def knn_regression(
        self,
        x: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Use the k-nearest neighbors of the input x to predict its regression label.
        The prediction will be the average value of the labels from the k neighbors.

        Args:
            x: 1D tensor [D]
            k: int, number of neighbors

        Returns:
            average value of labels from the k neighbors. Tensor of shape [1]
        """
        return torch.as_tensor(self.get_k_nearest_neighbor(x, k)[1].mean())  # Implement me:
        #raise NotImplementedError
