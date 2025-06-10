import torch


class TimeScheduler:
    """
    A class to generate different time schedules for training.
    """

    @classmethod
    def get_uniform_time_schedule(cls, batch_size: int, device: torch.device):
        """
        Generates a uniform time schedule for the given batch size.
        :param batch_size: The number of samples in the batch.
        :param device: The device to place the tensor on (CPU or GPU).
        :return: A tensor of shape (batch_size, 1) with random time steps.
        """
        time_steps = torch.rand(batch_size, 1, device=device, requires_grad=True)
        return time_steps

    @classmethod
    def get_cosine_time_schedule(cls, batch_size: int, device: torch.device):
        """
        Generates a cosine time schedule for the given batch size.
        :param batch_size: The number of samples in the batch.
        :param device: The device to place the tensor on (CPU or GPU).
        :return: A tensor of shape (batch_size, 1) with time steps sampled from a cosine schedule.
        """
        beta = torch.linspace(0, 1, steps=1000)
        cosine_schedule = 1 - torch.cos(beta * torch.pi) / 2
        time_steps = cosine_schedule[torch.randint(0, 1000, (batch_size, 1))]
        return time_steps.to(device)
