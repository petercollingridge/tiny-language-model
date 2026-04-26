const background = document.querySelector(`#${svgId} .background`);
const inputNodes = document.querySelectorAll(`#${svgId} .input-node`);
const nodeElements = document.querySelectorAll(`#${svgId} .node > *`);
const activationElements = document.querySelectorAll(`#${svgId} .activation-value`);
const edgeElements = document.querySelectorAll(`#${svgId} .edge line`);

let selectedNode = -1;

const deselectNodes = () => {
  selectedNode = -1;
  nodeElements.forEach(node => {
    node.classList.remove('active', 'deactive');
    node.style.opacity = 1;
  });
  edgeElements.forEach(edge => {
    edge.style.opacity = 1;
  });
};

const selectNode = (index) => {
  selectedNode = index;
  const activationPattern = activations[index];

  activationPattern.nodes.forEach((activation, i) => {
    const node = nodeElements[i];
    if (activation > 0) {
      node.classList.add('active');
      node.classList.remove('deactive');
      node.style.opacity = 0.2 + 0.8 * activation;
      activationElements[i].textContent = activation.toFixed(2);
    } else {
      node.classList.add('deactive');
      node.classList.remove('active');
      node.style.opacity = 1;
    }
  });


  activationPattern.edges.forEach((activation, i) => {
    const edge = edgeElements[i];
    edge.style.opacity = activation;
  });

};

background.addEventListener('click', deselectNodes);

inputNodes.forEach((node, index) => {
  node.addEventListener('click', () => {
    if (selectedNode === index) {
      deselectNodes();
    } else {
      selectNode(index);
    }
  });
});

