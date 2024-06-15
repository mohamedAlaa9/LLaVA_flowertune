from collections import OrderedDict
from typing import Callable, Dict, Tuple
import os
import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from trl import SFTTrainer
from transformers import TrainingArguments, LlavaForConditionalGeneration
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from models import get_model, cosine_annealing


# pylint: disable=too-many-arguments
class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        # formatting_prompts_func,
        data_collator,
        save_path,
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_arguments = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        # self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.save_path = save_path

        # instantiate model
        # self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype = torch.float16)
        self.model = get_model(model_cfg)
        self.trainset = trainset
        self.state_dict = None
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""

        state_dict = get_peft_model_state_dict(self.model)
        self.state_dict = state_dict
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.train_cfg.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = self.save_path
        self.tokenizer.padding_side = "right"
        # Construct trainer
        # trainer = SFTTrainer(
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     args=self.training_arguments,
        #     formatting_func=self.formatting_prompts_func,
        #     max_seq_length=self.train_cfg.seq_length,
        #     train_dataset=self.trainset,
        #     data_collator=self.data_collator,
        # )
        # training_args = TrainingArguments(
        #     output_dir="llava-1.5-7b-hf-ft-mix-vsft",
        #     report_to="tensorboard",
        #     learning_rate=1.4e-5,
        #     per_device_train_batch_size=8,
        #     gradient_accumulation_steps=1,
        #     logging_steps=5,
        #     num_train_epochs=1,
        #     # push_to_hub=True,
        #     gradient_checkpointing=True,
        #     remove_unused_columns=False,
        #     fp16=True,
        #     bf16=False
        # )
        # training_args.learning_rate = new_lr
        # training_args.output_dir = self.save_path
        trainer = SFTTrainer(
            model=self.model,
            args=self.training_arguments,
            train_dataset=self.trainset,
            dataset_text_field="text",  # need a dummy field
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            # formatting_func=self.formatting_prompts_func,
            max_seq_length=self.train_cfg.seq_length,
            dataset_kwargs={"skip_prepare_dataset": True},
        )
        # Do local training
        results = trainer.train()
        self.model.config.save_pretrained(self.training_arguments.output_dir)
        self.model.save_pretrained(self.training_arguments.output_dir, state_dict = self.state_dict)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters()
        )
        torch.save(non_lora_state_dict, os.path.join(self.training_arguments.output_dir, 'non_lora_trainables.bin'))
        return (
            self.get_parameters({}),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return
def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def gen_client_fn(
    fds,
    tokenizer,
    # formatting_prompts_func,
    data_collator,
    model_cfg: DictConfig,
    train_cfg: DictConfig,
    save_path: str,
    partition_id: int = 0,
    api: bool = False,
) -> Callable[[str], FlowerClient]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the Flower Clients."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Let's get the partition corresponding to the i-th client
        client_trainset = (
            fds.load_partition(partition_id, "train")
            if api
            else fds.load_partition(int(cid), "train")
        )
        # client_trainset = client_trainset.rename_column("output", "response")

        return FlowerClient(
            model_cfg,
            train_cfg,
            client_trainset,
            tokenizer,
            # formatting_prompts_func,
            data_collator,
            save_path,
        ).to_client()

    return client_fn