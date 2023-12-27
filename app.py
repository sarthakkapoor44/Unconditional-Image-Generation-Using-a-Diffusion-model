from imports import *
from data_preprocessing import *
from model_architecture import *
from inference import *

model = SimpleUnet()
st.title("Generating images using a diffusion model")
model.load_state_dict(torch.load("new_linear_model_1090.pt", map_location=torch.device('cpu')))

if (st.button("Click to generate image")):
    samples = sample(model, image_size=img_size, batch_size=64, channels=3)
    num_columns = 5
    for i in range(0, 60, num_columns):
        cols = st.columns(num_columns)
        for col, img_idx in zip(cols, range(i, i + num_columns)):
            reverse_transforms = transforms.Compose([
                transforms.Lambda(lambda t: (t + 1) / 2),
                transforms.Lambda(lambda t: t.permute(1, 2, 0)),
                transforms.Lambda(lambda t: t * 255.),
                transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
                transforms.ToPILImage(),
            ])
            img = reverse_transforms(torch.Tensor((samples[-1][img_idx].reshape(3, img_size, img_size))))
            col.image(img, width=150)
