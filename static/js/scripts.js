function scrollToContent(id, event) {
    const content = document.getElementById(id);
    const isVisible = content.getBoundingClientRect().top >= 0 && content.getBoundingClientRect().bottom <= window.innerHeight;
    
    if (!isVisible) {
        event.preventDefault(); 
        content.scrollIntoView({ behavior: 'smooth' });
    }
}

window.onscroll = function() { scrollFunction() };

function scrollFunction() {
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
        document.getElementById("scrollToTop").style.display = "block";
    } else {
        document.getElementById("scrollToTop").style.display = "none";
    }
}

function scrollToTop() {
    document.body.scrollTop = 0; // For Safari
    document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE, and Opera
}

document.addEventListener("DOMContentLoaded", function() {
    var collapseOne = document.getElementById("collapseOne");
    var accordionButton = document.querySelector('[data-bs-target="#collapseOne"]');
    if (collapseOne) {
        collapseOne.classList.remove("show");
    }
    if (accordionButton) {
        accordionButton.setAttribute("aria-expanded", "false");
    }

    
});

function handleNavLinkClick(event) {
    const links = document.querySelectorAll('.nav-link');
    links.forEach(link => {
        link.classList.remove('active');
    });
    event.currentTarget.classList.add('active');
}


var scrollSpy = new bootstrap.ScrollSpy(document.body, {
    target: '#sidebar'
});

  