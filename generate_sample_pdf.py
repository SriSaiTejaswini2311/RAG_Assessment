from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def create_sample_pdf(output_path: str):
    c = canvas.Canvas(output_path, pagesize=letter)
    
    # Page 1
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, 750, "Acme Corporation - Corporate Report")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 700, "1. Executive Summary")
    
    c.setFont("Helvetica", 12)
    text_lines = [
        "Welcome to the official company profile of the Acme Corporation.",
        "The Acme Corporation was founded in 2028 by the visionary technologist Jane Doe.",
        "Our mission is to pioneer advanced mobility and levitation systems for future generations.",
        "Currently, our primary focus is developing gravity-defying apparel.",
        "The flagship product of Acme is the 'Anti-Gravity Boots v4.2', which operates using quantum levitation.",
        "This revolutionary technology allows users to hover up to 2 meters above any solid surface.",
        "The retail price of Anti-Gravity Boots v4.2 is set at $12,499 per pair.",
        "The company headquarters is located in New Portland, on the moon."
    ]
    
    y = 660
    for line in text_lines:
        c.drawString(100, y, line)
        y -= 25
        
    c.showPage()
    
    # Page 2
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 700, "2. Technical Specifications")
    
    tech_lines = [
        "The Anti-Gravity Boots v4.2 feature a battery life of 12 hours on a single charge.",
        "Charging from 0% to 100% requires approximately 45 minutes using the proprietary Acme Supercharger.",
        "The maximum weight capacity supported by the quantum levitation engine is 150 kilograms.",
        "They are equipped with safety stabilizers that activate automatically in case of sudden drops."
    ]
    
    y = 660
    for line in tech_lines:
        c.drawString(100, y, line)
        y -= 25
        
    c.showPage()
    c.save()

if __name__ == "__main__":
    create_sample_pdf("sample.pdf")
    print("sample.pdf generated successfully.")
