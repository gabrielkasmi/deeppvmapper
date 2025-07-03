# DeepPVMapper GitHub Pages Setup

This document provides instructions for setting up and hosting the DeepPVMapper project page on GitHub Pages at `gabrielkasmi.github.io/deeppvmapper`.

## Current Setup

The project page is built using the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) and includes:

- **index.html** - Main project page with content about DeepPVMapper
- **static/css/style.css** - Modern, responsive styling
- **static/js/script.js** - Interactive features and animations
- **figs/flowchart.png** - Algorithm flowchart (already available)
- **.nojekyll** - Prevents Jekyll processing on GitHub Pages

## Setup Instructions

### 1. Create the gh-pages Branch

If you haven't already created the gh-pages branch, run these commands:

```bash
# Create and switch to gh-pages branch
git checkout -b gh-pages

# Add all files
git add .

# Commit the changes
git commit -m "Add academic project page for DeepPVMapper"

# Push the branch to GitHub
git push origin gh-pages
```

### 2. Configure GitHub Pages

1. Go to your repository on GitHub: `https://github.com/gabrielkasmi/deeppvmapper`
2. Click on **Settings** tab
3. Scroll down to **Pages** section (or click on **Pages** in the left sidebar)
4. Under **Source**, select **Deploy from a branch**
5. Choose **gh-pages** branch from the dropdown
6. Select **/(root)** folder
7. Click **Save**

### 3. Add Required Images

Before the page goes live, you need to add these images to the `static/images/` directory:

#### Favicon
- Create a `favicon.ico` file (16x16 or 32x32 pixels)
- Place it in `static/images/favicon.ico`
- This will be the small icon shown in browser tabs

#### Teaser Image
- Create a `teaser.jpg` file (1200x630 pixels recommended)
- Place it in `static/images/teaser.jpg`
- This image will be used when the page is shared on social media

### 4. Test Locally (Optional)

To test the page locally before deploying:

```bash
# Start a simple HTTP server
python -m http.server 8000

# Or using Node.js
npx serve .

# Then visit http://localhost:8000
```

### 5. Deploy

After adding the images:

```bash
# Add the new images
git add static/images/

# Commit the changes
git commit -m "Add favicon and teaser image"

# Push to GitHub
git push origin gh-pages
```

## Page Structure

The project page includes the following sections:

1. **Header** - Project title, author, and quick links
2. **Abstract** - Brief description of the work
3. **Teaser Image** - Algorithm flowchart
4. **Method** - Technical approach description
5. **Results** - Summary of achievements
6. **Citation** - BibTeX citation for academic use
7. **References** - Academic references
8. **Footer** - Copyright and license information

## Customization

### Content Updates

To update the content, edit the `index.html` file:

- **Title and subtitle**: Update the `<title>` tag and header section
- **Abstract**: Modify the text in the `.abstract` section
- **Method**: Update the description in the `.method` section
- **Results**: Modify the `.results` section
- **Citation**: Update the BibTeX in the `.citation` section
- **References**: Modify the `.references` section

### Styling Updates

To change the appearance, edit `static/css/style.css`:

- **Colors**: Modify the CSS variables and color values
- **Typography**: Change font sizes, weights, and families
- **Layout**: Adjust spacing, padding, and margins
- **Responsive design**: Modify media queries for different screen sizes

### Adding New Sections

To add new sections, follow this pattern in `index.html`:

```html
<section class="new-section">
    <div class="container">
        <h2>New Section Title</h2>
        <div class="new-section-content">
            <!-- Your content here -->
        </div>
    </div>
</section>
```

Then add corresponding CSS in `style.css`:

```css
.new-section {
    /* Your styles here */
}

.new-section-content {
    /* Your styles here */
}
```

## Troubleshooting

### Page Not Loading
- Check that the gh-pages branch exists and contains the files
- Verify GitHub Pages is enabled in repository settings
- Wait a few minutes for deployment (GitHub Pages can take 5-10 minutes)

### Images Not Showing
- Ensure image paths are correct (relative to the root directory)
- Check that images are actually committed and pushed to the gh-pages branch
- Verify image file names match exactly (case-sensitive)

### Styling Issues
- Check browser console for CSS errors
- Verify CSS file path is correct in `index.html`
- Test in different browsers to ensure compatibility

## Maintenance

### Regular Updates
- Keep the page content current with your research progress
- Update links to papers, code, and other resources
- Add new results or visualizations as they become available

### Performance Optimization
- Compress images before uploading (use tools like TinyPNG)
- Consider lazy loading for large images
- Monitor page load times and optimize as needed

## Resources

- [GitHub Pages Documentation](https://pages.github.com/)
- [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)
- [CSS Grid and Flexbox Guide](https://css-tricks.com/snippets/css/complete-guide-grid/)
- [Responsive Design Best Practices](https://developers.google.com/web/fundamentals/design-and-ux/responsive)

## Support

If you encounter issues:
1. Check the GitHub Pages documentation
2. Review the browser console for errors
3. Test with a simple HTML file first
4. Consider reaching out to the GitHub community

---

**Note**: This setup creates a professional academic project page that showcases your DeepPVMapper work effectively. The page will be accessible at `https://gabrielkasmi.github.io/deeppvmapper` once deployed. 