import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from denoisegan.models import DenoiseGenerator, UNetDiscriminatorSN, DinoProjectedDiscriminator, DualDiscriminator

console = Console()

def test_modules(device):
    console.print("\n[bold cyan]=== Testing DenoiseGan Modules ===[/bold cyan]\n")
    
    try:
        console.print("Testing [bold yellow]DenoiseGenerator[/bold yellow]...")
        G = DenoiseGenerator(channels=(48, 96, 192, 320, 448), use_checkpoint=False).to(device)
        dummy_in = torch.randn(2, 3, 256, 256, device=device)
        with torch.no_grad():
            dummy_out = G(dummy_in)
        
        assert dummy_out.shape == (2, 3, 256, 256), f"Shape mismatch: {dummy_out.shape}"
        console.print(f"  [bold green]✓ PASSED[/bold green] | Input: {tuple(dummy_in.shape)} -> Output: {tuple(dummy_out.shape)}")
    except Exception as e:
        console.print(f"  [bold red]✗ FAILED[/bold red] DenoiseGenerator: {e}", style="red")
        G = None

    try:
        console.print("Testing [bold yellow]UNetDiscriminatorSN[/bold yellow]...")
        D_unet = UNetDiscriminatorSN(in_channels=3, num_feat=64).to(device)
        dummy_in = torch.randn(2, 3, 256, 256, device=device)
        with torch.no_grad():
            dummy_out = D_unet(dummy_in)
        
        assert dummy_out.shape == (2, 1, 256, 256), f"Shape mismatch: {dummy_out.shape}"
        console.print(f"  [bold green]✓ PASSED[/bold green] | Input: {tuple(dummy_in.shape)} -> Output: {tuple(dummy_out.shape)}")
    except Exception as e:
        console.print(f"  [bold red]✗ FAILED[/bold red] UNetDiscriminatorSN: {e}", style="red")

    try:
        console.print("Testing [bold yellow]DinoProjectedDiscriminator[/bold yellow]...")
        D_dino = DinoProjectedDiscriminator(layers=(2, 5, 8, 11)).to(device)
        dummy_in = torch.randn(2, 3, 256, 256, device=device)
        with torch.no_grad():
            dummy_out = D_dino(dummy_in)
        
        assert isinstance(dummy_out, list) and len(dummy_out) == 4, "Output is not a list of 4 layers"
        for i, out in enumerate(dummy_out):
            assert out.shape == (2, 1, 18, 18), f"Layer {i} shape mismatch: {out.shape}"
        console.print(f"  [bold green]✓ PASSED[/bold green] | Input: {tuple(dummy_in.shape)} -> 4 Layer Outputs of shape: {tuple(dummy_out[0].shape)}")
    except Exception as e:
        console.print(f"  [bold red]✗ FAILED[/bold red] DinoProjectedDiscriminator: {e}. Note: dino requires internet to fetch pretrained weights.", style="red")

    try:
        console.print("Testing [bold yellow]DualDiscriminator[/bold yellow]...")
        D_dual = DualDiscriminator(num_feat=64).to(device)
        dummy_in = torch.randn(2, 3, 256, 256, device=device)
        with torch.no_grad():
            unet_logit, dino_logits = D_dual(dummy_in)
        
        assert unet_logit.shape == (2, 1, 256, 256), f"UNet logit shape mismatch: {unet_logit.shape}"
        assert len(dino_logits) == 4, f"Dino logits length mismatch: {len(dino_logits)}"
        console.print(f"  [bold green]✓ PASSED[/bold green] | Input: {tuple(dummy_in.shape)} -> UNet Logit: {tuple(unet_logit.shape)}, Dino Layers: {len(dino_logits)}")
    except Exception as e:
        console.print(f"  [bold red]✗ FAILED[/bold red] DualDiscriminator: {e}", style="red")
        
    return G

def get_submodule_name(name):
    parts = name.split('.')
    if len(parts) > 0:
        return parts[0]
    return "other"

def install_package(package_name):
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    except subprocess.CalledProcessError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--break-system-packages"])
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Failed to install {package_name}: {e}[/bold red]")
            raise e

def visualize_weights(model, ckpt_path):
    console.print(f"\n[bold green]Loading checkpoint for weight visualization: {ckpt_path}[/bold green]")
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        console.print(f"[bold red]Failed to load checkpoint file: {e}[/bold red]")
        return
        
    state_dict = checkpoint.get('G', checkpoint)
    
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            clean_state_dict[k[10:]] = v
        else:
            clean_state_dict[k] = v
            
    msg = model.load_state_dict(clean_state_dict, strict=False)
    console.print(f"Loaded weights with status: missing keys={len(msg.missing_keys)}, unexpected keys={len(msg.unexpected_keys)}")
    
    submodule_weights = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            sub = get_submodule_name(name)
            if sub not in submodule_weights:
                submodule_weights[sub] = []
            submodule_weights[sub].append(param.detach().cpu())
            
    table = Table(title="Generator Weights Summary by Module", header_style="bold cyan")
    table.add_column("Submodule", style="yellow")
    table.add_column("Tensors Count", justify="right")
    table.add_column("Total Params", justify="right")
    table.add_column("Mean Value", justify="right")
    table.add_column("Std Value", justify="right")
    table.add_column("Min Value", justify="right")
    table.add_column("Max Value", justify="right")
    
    for sub, tensors in sorted(submodule_weights.items()):
        flat = torch.cat([t.flatten() for t in tensors])
        total_p = flat.numel()
        mean_v = flat.mean().item()
        std_v = flat.std().item()
        min_v = flat.min().item()
        max_v = flat.max().item()
        
        table.add_row(
            sub,
            f"{len(tensors)}",
            f"{total_p:,}",
            f"{mean_v:.5f}",
            f"{std_v:.5f}",
            f"{min_v:.5f}",
            f"{max_v:.5f}"
        )
    
    console.print(table)
    
    console.print("Generating weight distribution plots...")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("matplotlib not found. Installing matplotlib...")
        install_package("matplotlib")
        import matplotlib.pyplot as plt
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Weight Distribution (Checkpoint: {os.path.basename(ckpt_path)})", fontsize=16, fontweight='bold')
    
    all_weights = torch.cat([p.flatten().detach().cpu() for p in model.parameters() if p.requires_grad and p.ndim > 1])
    
    axes[0, 0].hist(all_weights.numpy(), bins=100, color='royalblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title("Overall Weight Distribution")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    if hasattr(model, 'in_proj') and hasattr(model.in_proj, 'weight'):
        in_w = model.in_proj.weight.flatten().detach().cpu().numpy()
        axes[0, 1].hist(in_w, bins=50, color='darkorange', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title("Input Projection (in_proj) Weights")
        axes[0, 1].set_xlabel("Value")
        axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[0, 1].text(0.5, 0.5, "in_proj weights not found", ha='center', va='center')
        
    if hasattr(model, 'to_out') and hasattr(model.to_out, 'weight'):
        out_w = model.to_out.weight.flatten().detach().cpu().numpy()
        axes[1, 0].hist(out_w, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title("Output Projection (to_out) Weights")
        axes[1, 0].set_xlabel("Value")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[1, 0].text(0.5, 0.5, "to_out weights not found", ha='center', va='center')
        
    bottleneck_params = []
    for name, p in model.named_parameters():
        if 'bottleneck_layers' in name and 'weight' in name:
            bottleneck_params.append(p.flatten().detach().cpu())
            
    if bottleneck_params:
        btn_w = torch.cat(bottleneck_params).numpy()
        axes[1, 1].hist(btn_w, bins=50, color='crimson', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title("Bottleneck Layers Weights")
        axes[1, 1].set_xlabel("Value")
        axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[1, 1].text(0.5, 0.5, "bottleneck weights not found", ha='center', va='center')
        
    plt.tight_layout()
    plt.savefig("weight_visualization.png", dpi=150)
    plt.close()
    console.print("[bold green]Weight distribution plots saved to weight_visualization.png[/bold green]")

def generate_structure_torchvista(model, device):
    console.print("\n[bold yellow]No trained model found. Generating model structure using torchvista...[/bold yellow]")
    try:
        import torchvista
    except ImportError:
        console.print("torchvista is not installed. Installing torchvista...")
        install_package("torchvista")
        import torchvista

    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    
    console.print("Tracing DenoiseGenerator model...")
    try:
        torchvista.trace_model(
            model,
            dummy_input,
            export_format='html',
            export_path='generator_structure.html'
        )
        console.print("[bold green]Model structure visualization saved to generator_structure.html[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Failed to trace model with torchvista: {e}[/bold red]")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default=None, help='Path to a specific model checkpoint (.pt)')
    p.add_argument('--ckpt-dir', type=str, default='./checkpoints', help='Directory to search for checkpoints')
    p.add_argument('--visualize', action='store_true', help='Visualize model weights if checkpoint exists, otherwise draw structure')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"Using device: [bold]{device}[/bold]")
    
    G = test_modules(device)
    
    if args.visualize:
        if G is None:
            console.print("[bold red]Cannot run visualization because DenoiseGenerator failed to build.[/bold red]")
            return
            
        G.eval()
        
        ckpt_path = None
        if args.checkpoint:
            if os.path.exists(args.checkpoint):
                ckpt_path = args.checkpoint
            else:
                console.print(f"[bold red]Specified checkpoint '{args.checkpoint}' not found.[/bold red]")
        else:
            if os.path.exists(args.ckpt_dir):
                pts = [os.path.join(args.ckpt_dir, f) for f in os.listdir(args.ckpt_dir) if f.endswith('.pt')]
                if pts:
                    finals = [f for f in pts if 'final' in os.path.basename(f)]
                    if finals:
                        ckpt_path = sorted(finals)[-1]
                    else:
                        ckpt_path = sorted(pts)[-1]
                        
        if ckpt_path:
            visualize_weights(G, ckpt_path)
        else:
            generate_structure_torchvista(G, device)

if __name__ == '__main__':
    main()
