class SVG_Element:
    """ Generic element with attributes and potential child elements.
        Outputs as <tag attribute dict> child </tag>."""

    indent = 4

    def __init__(self, tag, attributes=None, child=None):
        self.tag = tag
        self.attributes = attributes or {}
        self.children = {'root': []}
        self.child_order = ['root']

        if 'attrs' in self.attributes:
            self.attributes.update(self.attributes['attrs'])
            del self.attributes['attrs']

        if child is not None:
            self.children['root'] = [str(child)]

    def _write_number(self, value):
        return format(value, ".1f").rstrip('0').rstrip('.')

    def add(self, tag, attributes=None, child=None):
        """
            Create an element with given tag and atrributes,
            and append to self.children.
            Returns the child element.
        """

        child = SVG_Element(tag, attributes, child)
        self.children['root'].append(child)
        return child
    
    def add_section(self, section_id):
        section = SVG_Section()
        self.children[section_id] = section
        self.child_order.append(section_id)
        return section

    def get_section(self, section_id):
        return self.children[section_id]

    def circle(self, x, y, r, **kwargs):
        kwargs['cx'] = self._write_number(x)
        kwargs['cy'] = self._write_number(y)
        kwargs['r'] = self._write_number(r)
        return self.add('circle', kwargs)

    def line(self, x1, y1, x2, y2, **kwargs):
        kwargs['x1'] = self._write_number(x1)
        kwargs['y1'] = self._write_number(y1)
        kwargs['x2'] = self._write_number(x2)
        kwargs['y2'] = self._write_number(y2)
        return self.add('line', kwargs)

    def rect(self, x, y, width, height, **kwargs):
        kwargs['x'] = self._write_number(x)
        kwargs['y'] = self._write_number(y)
        kwargs['width'] = self._write_number(width)
        kwargs['height'] = self._write_number(height)
        return self.add('rect', kwargs)

    def output(self, nesting=0):
        indent = ' ' * nesting * self.indent
        svg_string = f'{indent}<{self.tag}'

        for key, value in self.attributes.items():
            if key == 'classname':
                key = 'class'
            svg_string += f' {key}="{value}"'

        child_string, new_line = self._write_children(nesting + 1)
    
        if child_string:
            svg_string += '>' + child_string

            if new_line:
                svg_string += '\n' + indent

            svg_string += f'</{self.tag}>'

        else:
            # Self closing tag
            svg_string += '/>'

        return svg_string

    def _write_children(self, nesting):
        child_string = ''
        new_line = False

        for child_name in self.child_order:
            try:
                children = self.children[child_name]
            except KeyError:
                print(f'No child with name {child_name}')
                continue

            if isinstance(children, SVG_Section):
                section_string, new_line = children._write_children(nesting)
                child_string += section_string
            else:
                for child in children:
                    if isinstance(child, SVG_Element):
                        child_string += '\n' + child.output(nesting)
                        new_line = True
                    else:
                        child_string += child
    
        return child_string, new_line


class SVG(SVG_Element):
    """ SVG element with style element and output that includes XML document string. """

    def __init__(self, attributes=None):
        SVG_Element.__init__(self, 'svg', attributes)
        self.attributes['xmlns'] = 'http://www.w3.org/2000/svg'

        style_element = SVG_Style_Element()
        self.children['style'] = [style_element]
        self.child_order = ['style', 'root']

        self.style_dict = style_element.children

    def add_style(self, element, attributes):
        """
            Add style to element in self.style.children using a dictionary in
            form {selector: value}
        """

        if element not in self.style_dict:
            self.style_dict[element] = {}
        self.style_dict[element].update(attributes)

    def outputToFile(self, filename):
        """ Prints output to a given filename. Add a .svg extenstion if not given. """

        import os
        if not os.path.splitext(filename)[1] == '.svg':
            filename += '.svg'

        with open(filename, 'w') as f:
            f.write(self.output())

    def write(self, filename=None):
        """ Write output to file if given a filename, otherwise return output as a string. """

        if not filename:
            return self.output()
        else:
            self.outputToFile(filename)


class SVG_Style_Element(SVG_Element):
    def __init__(self):
        self.children = {}

    def output(self, nesting=0):
        if not self.children:
            return ''

        style_string = '\n<style>\n'

        for element, style in self.children.items():
            style_string += '  %s {\n' % element

            for key, value in style.items():
                style_string += f'    {key}: {value};\n'
            style_string += '  }\n'

        style_string += '  </style>\n'

        return style_string


class SVG_Section(SVG_Element):
    """
    An empty element which is not written but can be used to organise SVG elements.
    """

    def __init__(self):
        self.children = {'root': []}
        self.child_order = ['root']
