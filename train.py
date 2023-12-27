from imports import *
from data_preprocessing import *
from diffusion import *
from model_architecture import *

@torch.no_grad()
def sample_timestep(x, t):

    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:

        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(timesteps/num_images)

    for i in range(0,timesteps)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)

        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show()



data = load_dataset("nelorth/oxford-flowers", split="train")

transform = Compose([
        Resize([img_size, img_size]),
        CenterCrop(img_size),
        ToTensor(),
        Lambda(lambda t: (t * 2) - 1)
])

data_tensor = transform(data[0]['image']).unsqueeze_(0).to(device)

for i in tqdm(range(1, 1000)):
  x = transform(data[i]["image"]).unsqueeze_(0).to(device)
  data_tensor = torch.cat((data_tensor, x))

data_loader = torch.utils.data.DataLoader(dataset = data_tensor, batch_size=BATCH_SIZE, shuffle = True, drop_last  =True)

model = SimpleUnet()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 25
for epoch in tqdm(range(epochs)):
    for step, batch in tqdm(enumerate(data_loader)):
      optimizer.zero_grad()

      t = torch.randint(0, timesteps, (BATCH_SIZE,), device=device).long()
      loss = get_loss(model, batch, t)
      loss.backward()
      optimizer.step()

      if epoch % 5 == 0 and step == 0:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        sample_plot_image()