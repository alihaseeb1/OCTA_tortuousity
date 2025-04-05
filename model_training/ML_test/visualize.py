import matplotlib.pyplot as plt

def visualize_vessel(coordinates, curvatures, turn_indices, curvature_threshold=0.06):
    """Enhanced visualization with turn markers and curvature plot"""
    plt.figure(figsize=(12, 6))
    
    # Main vessel plot
    plt.subplot(1, 2, 1)
    plt.plot(coordinates[:, 0], coordinates[:, 1], 'k-', lw=2, alpha=0.3, label='Path')
    
    # Plot curvature heatmap
    if len(curvatures) > 0:
        valid_points = coordinates[1:-1]
        max_curv = max(abs(curvatures.min()), abs(curvatures.max()))
        vmax = max(0.1, max_curv)
        sc = plt.scatter(valid_points[:, 0], valid_points[:, 1], 
                        c=curvatures, cmap='coolwarm', 
                        s=50, vmin=-vmax, vmax=vmax,
                        label='Curvature')
        plt.colorbar(sc, label='Curvature (1/pixel)')
        
        # Mark turns
        if turn_indices:
            turn_points = valid_points[turn_indices]
            plt.scatter(turn_points[:, 0], turn_points[:, 1],
                       s=100, marker='X', c='lime', edgecolor='k',
                       label=f'Turns ({len(turn_indices)})')
    
    plt.title("Vessel Path Analysis")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.gca().set_aspect('equal')
    
    # Curvature profile plot
    plt.subplot(1, 2, 2)
    if len(curvatures) > 0:
        plt.plot(curvatures, 'b-', label='Curvature')
        plt.scatter(turn_indices, [curvatures[i] for i in turn_indices],
                   c='lime', edgecolor='k', s=80, marker='X',
                   label='Detected Turns')
        plt.axhline(curvature_threshold, color='r', linestyle='--', label='Threshold')
        plt.axhline(-curvature_threshold, color='r', linestyle='--')
        plt.title("Curvature Profile")
        plt.xlabel("Position along vessel")
        plt.ylabel("Curvature (1/pixel)")
        plt.legend()
    
    plt.tight_layout()
    plt.show()

