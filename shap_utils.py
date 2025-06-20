import matplotlib.pyplot as plt
import shap

def plot_summary_figures(shap_values, X_test):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title('SHAP Summary Bar Plot', fontsize=16)
    plt.tight_layout()
    plt.savefig('SHAP_Bar_Plot.pdf', bbox_inches='tight')
    plt.savefig('SHAP_Bar_Plot.png', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP Summary Dot Plot', fontsize=16)
    plt.tight_layout()
    plt.savefig('SHAP_Dot_Plot.pdf', bbox_inches='tight')
    plt.savefig('SHAP_Dot_Plot.png', bbox_inches='tight')
    plt.show()